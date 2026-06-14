from logging import getLogger
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from utils.misc import to_cuda, set_worker_sharing_strategy
from dataset import get_dataset
from data_utils.collate import custom_collate
from utils.muon import Muon
from collections import OrderedDict, defaultdict
from pathlib import Path
import wandb

logger = getLogger()


class Trainer(object):
    def __init__(self, modules, params, symbol_env):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.params = params
        self.symbol_env = symbol_env

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch

        # set parameters
        self.set_parameters()

        # distributed
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                if k.endswith("_ema"):
                    continue
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                    # find_unused_parameters=True,
                )

        # set optimizer
        self.set_optimizer()

        # amp
        self.scaler = torch.amp.GradScaler("cpu" if params.cpu else "cuda", enabled=False)  # no longer needed with bf16

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-np.inf if biggest else np.inf) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0

        # reload potential checkpoints
        self.reload_checkpoint()

        # create data loaders
        if not params.eval_only:
            self.dataloader_count = 0
            self.dataset = get_dataset(params, symbol_env, split="train")
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=params.batch_size,
                # shuffle=True,
                num_workers=params.num_workers,
                worker_init_fn=set_worker_sharing_strategy,
                drop_last=True,
                pin_memory=True,
                collate_fn=custom_collate(
                    params.data.max_output_dimension,
                    symbol_env.pad_index,
                    params.data.tie_fields,
                    self.params.data.get("mixed_length", 0),
                    params.input_len,
                    params.symbol.pad_right,
                ),
            )
            self.data_iter = iter(self.dataloader)

        self.train_stats = defaultdict(list)

        if not params.use_raw_time:
            self.input_len = params.input_len
            self.output_len = params.data.t_num - self.input_len
            self.t = torch.linspace(0, 10, params.data.t_num, dtype=torch.float32)[None]  # (1, t_num)

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            num = sum([torch.numel(p) for p in v])
            logger.info(f"Found {num:,} parameters in {k}.")
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params

        match params.optim.type:
            case "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.parameters["model"],
                    lr=params.optim.lr,
                    weight_decay=params.optim.weight_decay,
                    eps=params.optim.get("eps", 1e-8),
                    amsgrad=params.optim.get("amsgrad", False),
                    betas=(0.9, params.optim.get("beta2", 0.999)),
                )

            case "muon":
                named_params = []
                for v in self.modules.values():
                    named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])

                # parameters containing these will be sent to adam
                # adam_keys = ["embed"]
                # adam_keys = ["embedding", "in_proj", "head"]
                adam_keys = ["embedding", "encoder.conv_in", "decoder.conv_out"]

                muon_params, adam_params = [], []
                muon_param_count = adam_param_count = 0
                for n, p in named_params:
                    if p.requires_grad:
                        if p.ndim < 2 or any([s in n for s in adam_keys]):
                            adam_params.append(p)
                            adam_param_count += p.numel()

                            # if p.ndim >= 2:
                            #     logger.info(n)
                        else:
                            muon_params.append(p)
                            muon_param_count += p.numel()

                logger.info(f"Muon parameters: {muon_param_count:,}, Adam parameters: {adam_param_count:,}")

                self.optimizer = Muon(
                    lr=params.optim.lr,
                    wd=params.optim.weight_decay,
                    muon_params=muon_params,
                    adamw_params=adam_params,
                    adamw_betas=(0.9, params.optim.get("beta2", 0.95)),
                    adamw_eps=params.optim.get("eps", 1e-8),
                )

            case _:
                raise ValueError(f"Unknown optimizer type: {params.optim.type}")

        if params.optim.scheduler_type:
            scheduler_args = {}

            match params.optim.scheduler_type:
                case "cosine_with_restarts":
                    scheduler_args["num_cycles"] = params.optim.get("num_cycles", 1)

                case "cosine_with_min_lr":
                    if "min_lr" in params.optim:
                        scheduler_args["min_lr"] = params.optim.min_lr
                    if "min_lr_rate" in params.optim:
                        scheduler_args["min_lr_rate"] = params.optim.min_lr_rate

                case "warmup_stable_decay":
                    scheduler_args["num_decay_steps"] = int(params.optim.max_iters * params.optim.decay)
                    scheduler_args["min_lr_ratio"] = params.optim.get("min_lr_ratio", 0)
                    # scheduler_args["num_stable_steps"] = (
                    #     params.optim.max_iters - params.optim.warmup - scheduler_args["num_decay_steps"]
                    # )

            self.scheduler = get_scheduler(
                name=params.optim.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=params.optim.warmup,
                num_training_steps=params.optim.max_iters,
                scheduler_specific_kwargs=scheduler_args,
            )
        else:
            self.scheduler = None

        logger.info(f"Optimizer: {type(self.optimizer)}, scheduler: {type(self.scheduler)}")

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if torch.isnan(loss).any():
            logger.warning("NaN detected")
            raise ValueError("NaN detected")

        params = self.params
        train_log = OrderedDict()

        if params.accumulate_gradients > 1:
            loss = loss / params.accumulate_gradients
        train_log["data_loss"] = loss.item()

        # optimizer

        optimizer = self.optimizer
        self.scaler.scale(loss).backward()

        if (self.n_total_iter + 1) % params.accumulate_gradients == 0:
            if params.clip_grad_norm > 0:
                self.scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                train_log["grad_norm"] = grad_norm.item()
            self.scaler.step(optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if params.ema.enable:
                self.modules["model_ema"].update()

        return train_log

    def log_train_stats(self, train_log):
        """
        Print statistics about the training.
        """

        ## wandb logging
        if self.params.use_wandb:
            logs = {
                "step": self.n_total_iter,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_log,
            }
            wandb.log({"train": logs})

        ## log loss info
        for k, v in train_log.items():
            self.train_stats[k].append(v)
        if (self.params.log_periodic > 0) and (self.n_iter % self.params.log_periodic == 0):
            s = f"Epoch {self.epoch:<2d} | step {self.n_iter:<4d}"
            for k, v in self.train_stats.items():
                s += f" | {k} = {np.mean(v):.6f}"

            self.train_stats.clear()
            logger.info(s)

        ## log other info
        if (self.params.print_freq > 0) and self.n_total_iter % self.params.print_freq == 0:
            # iteration number
            s_iter = f"{self.n_total_iter:7d} - "

            # memory usage
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            s_mem = f" MEM: {max_mem:.2f} MB - "

            # learning rates
            s_lr = (" LR: ") + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)

            logger.info(s_iter + s_mem + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "dataloader_count": self.dataloader_count,
            "best_metrics": self.best_metrics,
        }

        for k, v in self.modules.items():
            data[k] = v.state_dict()

        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()

        path = str(self.params.dump_path / f"{name}.pth")
        torch.save(data, path)

        logger.info(f"Saved {name} to {path}.")

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"

        if self.params.reload_checkpoint is not None:
            checkpoint_path = self.params.reload_checkpoint
            if not checkpoint_path.endswith(".pth"):
                checkpoint_path = str(Path(self.params.reload_checkpoint) / path)
            assert Path(checkpoint_path).exists()
        else:
            if root is not None:
                checkpoint_path = str(Path(root) / path)
            else:
                checkpoint_path = str(self.params.dump_path / path)
            if not Path(checkpoint_path).exists():
                logger.warning("Checkpoint path does not exist, {}".format(checkpoint_path))
                return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # reload model parameters
        for k, v in self.modules.items():
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            # v.requires_grad = requires_grad

        # reload optimizer
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        if "scaler" in data and self.scaler is not None:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])

        if "scheduler" in data and self.scheduler is not None:
            logger.warning("Reloading scheduler...")
            self.scheduler.load_state_dict(data["scheduler"])

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.dataloader_count = data["dataloader_count"]
        self.best_metrics = data["best_metrics"]
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None:
                _metric = prefix + "_" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info(f"New best score for {metric}: {scores[_metric]:.6f}")
                self.save_checkpoint(f"best-{metric}")

    def get_batch(self):
        """
        Return a training batch
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.dataloader_count += 1
            logger.info(f"Reached end of dataloader, restart {self.dataloader_count}...")
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    def diffusion_loss_fn(self, output_d, data_mask):
        """
        input shape: (b, c, x, y)
            NOTE: ignore mask for now
        """
        match self.params.model.scheduler.prediction_type:
            case "epsilon":
                output, noise = output_d["output"], output_d["noise"]
                loss = F.mse_loss(output.float(), noise.float())  # this could have different weights!
            case "sample":
                output, label, weights = output_d["output"], output_d["label"], output_d["snr_weights"]
                loss = weights * F.mse_loss(output.float(), label.float(), reduction="none")
                loss = loss.mean()
            case _:
                raise ValueError(f"Unsupported prediction type: {self.params.model.prediction_type}")
        return loss

    def data_loss_fn(self, data_output, data_label, data_mask):
        """
        data_output/data_label: Tensor (bs, output_len, x_num, x_num, dim)
        """
        # prepare weights for loss function
        if self.params.loss_weight == "l2":
            weight = torch.linalg.vector_norm(data_label, dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        elif self.params.loss_weight == "linfty":
            weight, _ = torch.max(torch.abs(data_label), dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        else:
            weight = None

        if weight is None:
            # no re-weighting, just regular MSE
            loss = F.mse_loss(data_output, data_label, reduction="none")
            loss = loss.sum() / torch.count_nonzero(data_mask.expand_as(loss))
        else:
            # reweight by weight
            eps = 1e-6
            if self.params.square_loss:
                loss = F.mse_loss(data_output, data_label, reduction="none")
                loss = (loss / (weight**2 + eps)).sum() / data_label.size(0)
            else:
                loss = torch.linalg.vector_norm(data_output - data_label, dim=(2, 3), keepdim=True)
                loss = (loss / (weight + eps)).sum() / data_label.size(0)

        return loss

    def normalize_data(self, data_input, data_label=None):
        if self.params.normalize:
            eps = 1e-8
            if self.params.normalize == "meanvar":
                mean = torch.mean(data_input, axis=(1, 2, 3), keepdim=True)  # (bs, 1, 1, 1, dim)
                std = torch.std(data_input, axis=(1, 2, 3), keepdim=True) + eps  # (bs, 1, 1, 1, dim)
            elif self.params.normalize == "range":
                max = torch.amax(data_input, dim=(1, 2, 3), keepdim=True)
                min = torch.amin(data_input, dim=(1, 2, 3), keepdim=True)
                mean = (max + min) / 2
                std = (max - min) / 2 + eps
            elif self.params.normalize == "meanvar_c":
                mean = torch.mean(data_input, axis=(1, 2, 3, 4), keepdim=True)  # (bs, 1, 1, 1, 1)
                std = torch.std(data_input, axis=(1, 2, 3, 4), keepdim=True) + eps  # (bs, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown normalization method: {self.params.normalize}")

            data_input = (data_input - mean) / std

            if not self.params.denormalize_for_loss and data_label is not None:
                # compute loss in normalized space
                data_label = (data_label - mean) / std  # use same mean and std

        else:
            mean = 0
            std = 1

        return data_input, data_label, mean, std

    def prepare_data(self, samples, train=True):
        """
        Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)

        samples: data:         Tensor     (bs, max_len, x_num, x_num, dim)
                 data_mask:    BoolTensor (bs, 1/output_len, 1, 1, dim)
                 t:            Tensor     (bs, max_len)

        """

        model_input = {}

        data = samples["data"]
        data_mask = samples["data_mask"]  # (bs, 1/output_len, 1, 1, dim)

        if self.params.use_raw_time:
            t = samples["t"]
        else:
            t = self.t

        input_len = self.params.input_len
        data_input = data[:, :input_len]  # (bs, input_len, x_num, x_num, dim)

        if self.params.model.name.endswith("auto"):
            # prepare inputs for autoregressive training
            data_label = data[:, input_len:]  # (bs, output_len, x_num, x_num, dim)
            data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)

            data_input, data_label, mean, std = self.normalize_data(data_input, data_label)

            if train:
                model_input["data"] = torch.cat([data_input, data_label], dim=1)  # (bs, t_num, x_num, x_num, dim)

                model_input["input_len"] = self.params.get("loss_start_idx", input_len)
                data_label = model_input["data"][:, model_input["input_len"] :]

                t_num = data_mask.size(1)  # (bs, 1/output_len, 1, 1, dim)
                if t_num > 1 and (input_len - model_input["input_len"]) > 0:
                    # mixed length inputs
                    input_mask = data_mask[:, :1].expand(
                        -1, input_len - model_input["input_len"], -1, -1, -1
                    )  # (bs, input_len-1, 1, 1, dim)
                    data_mask = torch.cat([input_mask, data_mask], dim=1)  # (bs, t_num, 1, 1, dim)

                # add autoregressive cumulative noise on the label part
                if self.params.noise > 0 and self.params.noise_type == "cumulative":
                    data_label = data_label.clone()
                    noise = (
                        self.params.noise
                        * torch.cumsum(torch.sum(data_label**2, dim=(2, 3), keepdim=True), dim=1) ** 0.5
                        * torch.randn_like(data_label)
                    )
                    # noise = (
                    #     self.params.noise
                    #     * torch.sum(data_label**2, dim=(1, 2, 3), keepdim=True) ** 0.5
                    #     * torch.randn_like(data_label)
                    # )
                    if t_num > 1:
                        # avoid noise in padding locations for mixed length inputs
                        noise = noise * data_mask
                    model_input["data"][:, model_input["input_len"] :] += noise

            else:
                model_input["data_input"] = data_input
                # during testing, equations are the same in one batch
                model_input["data_mask"] = data_mask[:1, :1, :, :, :]  # (1, 1, 1, 1, dim)
                model_input["input_len"] = input_len

            model_input["times"] = to_cuda(t[..., None])  # (1, t_num, 1)

        elif self.params.model.name.startswith("diffusion"):
            data_label = data[:, input_len:]  # (bs, output_len, x_num, x_num, dim)
            data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)

            data_input, data_label, mean, std = self.normalize_data(data_input, data_label)

            model_input["data_input"] = data_input
            if train:
                model_input["data_label"] = data_label
            else:
                model_input["data_mask"] = data_mask[:1, :1, :, :, :]  # (1, 1, 1, 1, dim)

        else:
            # prepare inputs for operator / 1 step training

            data_label = data[:, input_len:]  # (bs, output_len, x_num, x_num, dim)
            data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)

            data_input, data_label, mean, std = self.normalize_data(data_input, data_label)

            input_times = t[:, :input_len, None]  # (bs, input_len, 1)
            output_times = (
                t[:, input_len:, None] - input_times[:, -1:]
            )  # (bs, output_len, 1), force a Markovian time stepping

            model_input["input_times"] = to_cuda(input_times)
            model_input["output_times"] = to_cuda(output_times)
            model_input["data_input"] = data_input
            model_input["data_mask"] = data_mask

        d = {
            "data_label": data_label,
            "data_mask": data_mask,
            "mean": mean,
            "std": std,
        }

        if "symbol_input" in samples:
            model_input["symbol_input"] = to_cuda(samples["symbol_input"])  # LongTensor (bs, symbol_len)
            model_input["symbol_padding_mask"] = to_cuda(samples["symbol_mask"])  # BoolTensor (bs, symbol_len)

        return model_input, d

    def iter(self):
        """
        One training step.
        """
        params = self.params

        samples = self.get_batch()

        model = self.modules["model"]
        model.train()

        # prepare data part

        model_input, d = self.prepare_data(samples)
        # model_input contains model input args, d contains other attributes

        # forward / loss

        """
        Model input: 
            check prepare_data() function

        Model output:
            data_output:  (bs, output_len, x_num, x_num, data_dim)
        """

        with torch.amp.autocast("cpu" if params.cpu else "cuda", enabled=bool(params.amp), dtype=torch.bfloat16):
            if self.params.model.name.startswith("diffusion"):
                output_d = model("fwd", **model_input)
                data_loss = self.diffusion_loss_fn(output_d, d["data_mask"])
            else:
                data_output = model("fwd", **model_input)  # (bs, output_len, x_num, x_num, data_dim)

                if self.params.normalize and self.params.denormalize_for_loss:
                    # denormalize data, compute loss in original space
                    data_output = data_output * d["std"] + d["mean"]

                data_output = data_output * d["data_mask"]
                data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])

        # optimize
        train_log = self.optimize(data_loss)
        return train_log
