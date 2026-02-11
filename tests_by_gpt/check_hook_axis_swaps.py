import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_bcat import MultiscaleBCAT  # noqa: E402


class StubTransformer(torch.nn.Module):
    def __init__(self, dim: int, spatial_tokens: int) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_tokens = spatial_tokens

    def forward(self, data, times=None, **kwargs):
        b, t, _, _, _ = data.shape
        data_len = t * self.spatial_tokens
        tokens = torch.zeros(b, data_len, self.dim, device=data.device, dtype=data.dtype)
        spatial = torch.arange(self.spatial_tokens, device=data.device).repeat(t)
        time = torch.arange(t, device=data.device).repeat_interleave(self.spatial_tokens)
        tokens[:, :, 0] = spatial
        tokens[:, :, 1] = time
        return tokens


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    model = MultiscaleBCAT(model_cfg, x_num=x_num, max_output_dim=max_output_dim)

    spatial_tokens = model_cfg.embedder.patch_num ** 2
    model.transformer = StubTransformer(model_cfg.dim_emb, spatial_tokens)

    hook_called = {"value": False}

    def pre_decode_hook(_module, inputs):
        hook_called["value"] = True
        data_output = inputs[0]
        b, total, _ = data_output.shape
        if total % spatial_tokens != 0:
            raise AssertionError("Pre-decode length not divisible by spatial_tokens.")
        t = total // spatial_tokens
        spatial = data_output[0, :, 0].view(t, spatial_tokens).cpu()
        time = data_output[0, :, 1].view(t, spatial_tokens).cpu()
        expected_spatial = torch.arange(spatial_tokens).repeat(t).view(t, spatial_tokens)
        expected_time = torch.arange(t).repeat_interleave(spatial_tokens).view(t, spatial_tokens)
        if not torch.equal(spatial, expected_spatial):
            raise AssertionError("Pre-decode spatial ordering is not time-major.")
        if not torch.equal(time, expected_time):
            raise AssertionError("Pre-decode time ordering is not time-major.")

    handle = model.embedder.post_proj.register_forward_pre_hook(pre_decode_hook)

    batch_size = 1
    t_total = 3
    data = torch.zeros(batch_size, t_total, x_num, x_num, max_output_dim)
    times = torch.arange(t_total, dtype=torch.float32).view(1, t_total, 1)

    model.fwd(data=data, times=times, input_len=1)
    handle.remove()

    if not hook_called["value"]:
        raise AssertionError("Pre-decode hook did not run.")

    print("Hook-based axis swap check passed.")


if __name__ == "__main__":
    main()
