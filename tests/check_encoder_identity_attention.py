import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import TwoLevelTransformerEncoderLayer  # noqa: E402


class FirstTokenPool(torch.nn.Module):
    def __init__(self, rate: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.rate = rate
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        return x_blocks[..., 0, : self.out_dim]


def _set_identity_mha(mha: torch.nn.Module) -> None:
    embed_dim = mha.embed_dim
    eye = torch.eye(embed_dim)
    for layer in (mha.linear_q, mha.linear_k, mha.linear_v, mha.out_proj):
        layer.weight.data.copy_(eye)
        if layer.bias is not None:
            layer.bias.data.zero_()


def main() -> None:
    torch.manual_seed(0)
    rate = 1
    spatial_tokens = 9
    time_len = 3
    fast_dim = 4
    slow_dim = 4
    seq_len = time_len * spatial_tokens

    layer = TwoLevelTransformerEncoderLayer(
        fast_embed_dim=fast_dim,
        slow_embed_dim=slow_dim,
        num_heads=2,
        rate=rate,
        pool_ffn=FirstTokenPool(rate=rate, in_dim=fast_dim, out_dim=slow_dim // 2),
        lift_ffn=torch.nn.Linear(slow_dim, fast_dim // 2, bias=False),
        flex_attn=False,
    )

    _set_identity_mha(layer.fast_mixer.self_branch.mha)
    _set_identity_mha(layer.fast_mixer.cross_branch.mha)
    _set_identity_mha(layer.slow_mixer.self_branch.mha)
    _set_identity_mha(layer.slow_mixer.cross_branch.mha)

    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    x_fast = torch.stack([t, s, t, s], dim=-1).float().unsqueeze(0)
    x_slow = x_fast.clone()

    attn_mask = torch.full((seq_len, seq_len), float("-inf"))
    attn_mask.fill_diagonal_(0.0)

    z_fast, z_slow = layer(
        x_fast,
        x_slow,
        time_dim=1,
        fast_attn_mask=attn_mask,
        is_causal=False,
        spatial_tokens=spatial_tokens,
    )

    if not torch.equal(z_fast[:, :, : fast_dim // 2], x_fast[:, :, : fast_dim // 2]):
        raise AssertionError("Identity attention layer did not preserve fast self tokens.")
    if not torch.equal(z_slow[:, :, : slow_dim // 2], x_slow[:, :, : slow_dim // 2]):
        raise AssertionError("Identity attention layer did not preserve slow self tokens.")

    print("Encoder identity attention check passed.")


if __name__ == "__main__":
    main()
