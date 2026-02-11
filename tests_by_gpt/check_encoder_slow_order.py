import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import (  # noqa: E402
    RecombineDecoder,
    SplitEncoder,
    TwoLevelTransformerEncoder,
    TwoScaleTransformerEncoderLayer,
)


class IdentityEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class FirstTokenPool(torch.nn.Module):
    def __init__(self, rate: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.rate = rate
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        return x_blocks[..., 0, : self.out_dim]


class IdentityMixer(torch.nn.Module):
    def forward(self, x, y=None, **kwargs):
        return x


def _make_fast_tokens(time_len: int, spatial_tokens: int, dim: int) -> torch.Tensor:
    total = time_len * spatial_tokens
    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    tokens = torch.zeros(1, total, dim)
    tokens[0, :, 0] = t
    tokens[0, :, 1] = s
    return tokens


def main() -> None:
    rate = 4
    spatial_tokens = 9
    time_len_fast = 5
    fast_dim = 8
    slow_dim = 8

    x_fast = _make_fast_tokens(time_len_fast, spatial_tokens, fast_dim)
    split_encoder = SplitEncoder(
        encoder=IdentityEncoder(),
        rate=rate,
        pool_ffn=FirstTokenPool(rate=rate, in_dim=fast_dim, out_dim=slow_dim),
        time_dim=1,
        spatial_tokens=spatial_tokens,
    )
    x_fast_out, x_slow = split_encoder(x_fast)
    assert torch.equal(x_fast_out, x_fast)

    slow_len = x_slow.size(1) // spatial_tokens
    x_slow = x_slow.view(1, slow_len, spatial_tokens, slow_dim)

    for k in range(slow_len):
        if k == 0:
            assert torch.all(x_slow[:, k] == 0)
            continue
        expected_time = (k - 1) * rate
        assert torch.all(x_slow[:, k, :, 0] == expected_time)
        expected_spatial = torch.arange(spatial_tokens).view(1, spatial_tokens)
        assert torch.equal(x_slow[:, k, :, 1], expected_spatial)

    layer = TwoScaleTransformerEncoderLayer(
        fast_embed_dim=fast_dim,
        slow_embed_dim=slow_dim,
        num_heads=2,
        rate=rate,
        pool_ffn=FirstTokenPool(rate=rate, in_dim=fast_dim, out_dim=slow_dim // 2),
        lift_ffn=torch.nn.Linear(slow_dim, fast_dim // 2, bias=False),
        flex_attn=False,
    )
    layer.fast_mixer = IdentityMixer()
    layer.slow_mixer = IdentityMixer()
    encoder = TwoLevelTransformerEncoder(layer, num_layers=1, norm_fast=None, norm_slow=None)

    z_fast, z_slow = encoder(
        x_fast_out,
        x_slow.view(1, -1, slow_dim),
        spatial_tokens=spatial_tokens,
        full=False,
    )
    assert torch.equal(z_slow, x_slow.view(1, -1, slow_dim))

    print("Encoder slow order check passed.")


if __name__ == "__main__":
    main()
