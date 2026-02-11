import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.embedder import get_embedder  # noqa: E402
from models.multiscale_utils import lift, pad_slow, pool  # noqa: E402


class FirstTokenPool(torch.nn.Module):
    def __init__(self, rate: int, in_dim: int) -> None:
        super().__init__()
        self.rate = rate
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        return x_blocks[..., 0, :]


def _make_fast_tokens(time_len: int, spatial_tokens: int) -> torch.Tensor:
    total = time_len * spatial_tokens
    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    return torch.stack([t, s], dim=-1).float().unsqueeze(0)


def _check_pad_slow(time_len: int, spatial_tokens: int, rate: int) -> None:
    x = _make_fast_tokens(time_len, spatial_tokens)
    padded, pad_steps = pad_slow(x, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    if pad_steps == 0:
        assert torch.equal(padded, x)
        return
    prefix = padded[:, : x.size(1)]
    tail = padded[:, x.size(1) :]
    assert torch.equal(prefix, x)
    assert torch.all(tail == 0)


def _check_lift(time_len: int, spatial_tokens: int, rate: int) -> None:
    slow = _make_fast_tokens(time_len, spatial_tokens)
    lifted = lift(slow, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    t_expected = torch.arange(time_len).repeat_interleave(rate).repeat_interleave(spatial_tokens)
    s_expected = torch.arange(spatial_tokens).repeat(time_len * rate)
    expected = torch.stack([t_expected, s_expected], dim=-1).float().unsqueeze(0)
    assert torch.equal(lifted, expected)


def _check_pool(time_len: int, spatial_tokens: int, rate: int) -> None:
    x = _make_fast_tokens(time_len, spatial_tokens)
    pool_ffn = FirstTokenPool(rate=rate, in_dim=x.size(-1))
    pooled = pool(x, rate=rate, time_dim=1, pool_ffn=pool_ffn, spatial_tokens=spatial_tokens)

    pooled = pooled.view(1, -1, spatial_tokens, x.size(-1))

    num_blocks = time_len // rate
    expected_len = num_blocks
    assert pooled.size(1) == expected_len

    for k in range(expected_len):
        if k == 0:
            assert torch.all(pooled[:, k] == 0)
            continue
        expected_time = (k - 1) * rate
        assert torch.all(pooled[:, k, :, 0] == expected_time)
        expected_spatial = torch.arange(spatial_tokens).view(1, spatial_tokens)
        assert torch.equal(pooled[:, k, :, 1], expected_spatial)


def _check_decode_rearrange(embedder, time_len: int) -> None:
    patch_num = embedder.config.patch_num_output
    spatial_tokens = patch_num ** 2
    dim = embedder.config.dim
    tokens = torch.zeros(1, time_len * spatial_tokens, dim)
    spatial = torch.arange(spatial_tokens).repeat(time_len)
    time = torch.arange(time_len).repeat_interleave(spatial_tokens)
    tokens[0, :, 0] = spatial
    tokens[0, :, 1] = time

    rearrange_layer = embedder.post_proj[0]
    out = rearrange_layer(tokens)
    out = out.view(1, time_len, dim, patch_num, patch_num)

    expected_spatial = torch.arange(spatial_tokens).view(patch_num, patch_num)
    for t in range(time_len):
        if not torch.equal(out[0, t, 0], expected_spatial):
            raise AssertionError("decode rearrange spatial order mismatch")
        if not torch.all(out[0, t, 1] == t):
            raise AssertionError("decode rearrange time order mismatch")


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    embedder = get_embedder(model_cfg.embedder, x_num, max_output_dim)

    spatial_tokens = model_cfg.embedder.patch_num ** 2
    rate = int(model_cfg.rate)

    for time_len in (4, 5, 8):
        _check_pad_slow(time_len, spatial_tokens, rate)
        _check_lift((time_len + rate - 1) // rate, spatial_tokens, rate)

    _check_pool(time_len=8, spatial_tokens=spatial_tokens, rate=rate)
    _check_decode_rearrange(embedder, time_len=3)

    print("Axis swap checks passed.")


if __name__ == "__main__":
    main()
