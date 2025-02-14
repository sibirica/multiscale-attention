from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
from utils.misc import to_cuda

logger = getLogger()

from trainer import Trainer


class VQTrainer(Trainer):
    def __init__(self, modules, params, symbol_env):
        super().__init__(modules, params, symbol_env)
        self.commit_loss = 0.0
        self.commit_loss_step = 0.0

    def prepare_data(self, samples, train=True):
        """
        Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)

        samples: data:         Tensor     (bs, max_len, x_num, x_num, dim)
                 data_mask:    BoolTensor (bs, 1/output_len, 1, 1, dim)

        """

        model_input = {}

        data = samples["data"]
        data_mask = samples["data_mask"]  # (bs, 1/output_len, 1, 1, dim)

        input_len = self.params.input_len
        data_input = data[:, :input_len]  # (bs, input_len, x_num, x_num, dim)
        data_label = data[:, input_len:]  # (bs, output_len, x_num, x_num, dim)
        data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)

        data_input, data_label, mean, std = self.normalize_data(data_input, data_label)
        model_input["input"] = torch.cat([data_input, data_label], dim=1)  # (bs, t_num, x_num, x_num, dim)

        d = {
            "data_label": model_input["input"].detach().clone(),
            "data_mask": data_mask,  # NOTE: modify for mixed length training
            "mean": mean,
            "std": std,
        }

        return model_input, d

    def loss_fn(self, output, target, mask):
        output = output * mask
        loss = F.mse_loss(output, target, reduction="none")
        loss = loss.sum() / torch.count_nonzero(mask.expand_as(loss))
        return loss

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
            output, diff = model("fwd", **model_input)
            data_loss = self.loss_fn(output, d["data_label"], d["data_mask"])

        self.data_loss_step = data_loss.item()
        self.data_loss += self.data_loss_step
        self.commit_loss_step = diff.item()
        self.commit_loss += self.commit_loss_step

        # optimize
        self.optimize(data_loss + diff)

        self.inner_epoch += 1
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()
