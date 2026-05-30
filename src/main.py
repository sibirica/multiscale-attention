import wandb
from pathlib import Path
import torch
import torch.multiprocessing

import utils
from utils.mode import init_distributed_mode
from utils.misc import initialize_exp, set_seed
from models.build_model import build_model
from symbol_utils.environment import SymbolicEnvironment
from omegaconf import DictConfig, OmegaConf
import hydra

from trainer import Trainer
from evaluate import Evaluator, metric_to_header


torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(params: DictConfig):
    if params.dryrun:
        print("---------------------------------")
        print("----------- Debug Run -----------")
        print("---------------------------------")
        params.max_epoch = 1
        params.n_steps_per_epoch = int(params.dryrun)
        params.exp_name = "debug"
        params.use_wandb = 0
        params.log_periodic = 1
        params.eval_size = 1
        params.base_seed = 1
        params.log_eval_plots = -1

    if params.eval_only:
        assert params.reload_model
        reload_model = Path(params.reload_model)
        if (reload_model / f"best-{params.validation_metrics}.pth").exists():
            params.reload_model = str(reload_model / f"best-{params.validation_metrics}.pth")
        elif (reload_model / "checkpoint.pth").exists():
            params.reload_model = str(reload_model / "checkpoint.pth")
        else:
            assert reload_model.is_file()

        if params.overfit_test and params.exp_id:
            params.exp_id = params.exp_id + "_train"

        if params.eval_single_file and params.exp_id:
            params.exp_id = params.exp_id + "_file"

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    utils.misc.CUDA = not params.cpu

    if "warmup" in params.optim:
        params.optim.max_iters = params.max_epoch * params.n_steps_per_epoch // params.accumulate_gradients
        if params.optim.warmup is not None and params.optim.warmup < 1:
            params.optim.warmup = max(1, int(params.optim.warmup * params.optim.max_iters))

    # initialize experiment / logger / config
    logger = initialize_exp(params)

    # wandb logging
    if not params.is_master:
        params.use_wandb = False
    if params.use_wandb:
        if not params.wandb.id:
            params.wandb.id = wandb.util.generate_id()
        if not params.wandb.name:
            params.wandb.name = params.exp_id
        wandb.init(
            project=params.wandb.project if params.wandb.project else params.exp_name,
            resume="allow",
            id=params.wandb.id,
            name=params.wandb.name,
            entity=params.wandb.entity,
            notes=params.wandb.notes,
            dir=str(params.dump_path),
        )

        # log configs on wandb, convert to dict
        config_d = OmegaConf.to_container(params, resolve=True, throw_on_missing=True)
        config = {"params": {}}
        keys_to_separate = ["model", "data", "optim", "wandb", "symbol"]
        for k, v in config_d.items():
            if k in keys_to_separate:
                config[k] = v
            else:
                config["params"][k] = v

        wandb.config.update(config, allow_val_change=True)

    # set seed for reproducibility
    if params.eval_only:
        set_seed(params.test_seed)
    else:
        set_seed(params.base_seed)

    # build model / trainer / evaluator

    symbol_env = SymbolicEnvironment(params.symbol)
    modules = build_model(params, params.model, params.data, symbol_env)

    if params.use_wandb and params.wandb.watch:
        wandb.watch(modules["model"], log="all")

    trainer = Trainer(modules, params, symbol_env)
    evaluator = Evaluator(trainer, symbol_env)

    if params.eval_only:
        stats, _ = evaluator.evaluate()

        s = "Eval | data loss = {:.6f}".format(stats["data_loss"])
        for metric in evaluator.validation_metrics:
            s += " | {} = {:.6f}".format(metric_to_header[metric], stats[metric])
        logger.info(s)

        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        logger.info("MEM: {:.2f} MB".format(max_mem))
        return

    while trainer.epoch < params.max_epoch:
        logger.info(f"============ Starting epoch {trainer.epoch} ... ============")

        trainer.n_iter = 0
        while trainer.n_iter < trainer.n_steps_per_epoch:
            train_log = trainer.iter()
            trainer.n_iter += 1
            trainer.n_total_iter += 1
            trainer.log_train_stats(train_log)

        logger.info(f"============ End of epoch {trainer.epoch} ============")

        logger.info("====== STARTING EVALUATION (multi-gpu: {}) =======".format(params.multi_gpu))
        stats, results_per_type = evaluator.evaluate()

        s = "Epoch {} Eval | data loss = {:.6f}".format(trainer.epoch, stats["data_loss"])
        for metric in evaluator.validation_metrics:
            s += " | {} = {:.6f}".format(metric_to_header[metric], stats[metric])
        logger.info(s)

        if params.use_wandb:
            stats["epoch"] = trainer.epoch
            wandb_log = {"val": {k.strip("_"): v for k, v in stats.items()}}
            if params.wandb.log_per_type:
                for type, results in results_per_type.items():
                    wandb_log["val"][type] = {
                        k.strip("_"): v for k, v in results.items() if k in ["_l2_error", "data_loss"]
                    }
            wandb.log(wandb_log)

        if params.is_master:
            trainer.save_checkpoint("checkpoint")
            trainer.save_best_model(stats)

            if params.save_periodic > 0 and (trainer.epoch + 1) % params.save_periodic == 0:
                trainer.save_checkpoint(f"periodic-{trainer.epoch}")

        trainer.epoch += 1

    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    logger.info("MEM: {:.2f} MB".format(max_mem))

    if params.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
