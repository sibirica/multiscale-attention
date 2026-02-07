import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import MixedSplitAttention, lift  # noqa: E402


def _alignment(tokens: torch.Tensor, spatial_tokens: int) -> tuple[float, float]:
    total = tokens.size(1)
    t_len = total // spatial_tokens
    t_major = torch.arange(total) // spatial_tokens
    s_major = torch.arange(total) % t_len
    t_acc = (tokens[0, :, 0].long() == t_major).float().mean().item()
    s_acc = (tokens[0, :, 0].long() == s_major).float().mean().item()
    return t_acc, s_acc


def _set_identity_mha(mha: torch.nn.Module) -> None:
    embed_dim = mha.embed_dim
    eye = torch.eye(embed_dim)
    for layer in (mha.linear_q, mha.linear_k, mha.linear_v, mha.out_proj):
        layer.weight.data.copy_(eye)
        if layer.bias is not None:
            layer.bias.data.zero_()


def main() -> None:
    torch.manual_seed(0)
    spatial_tokens = 9
    rate = 4
    t_fast = 5
    t_slow = (t_fast + rate - 1) // rate
    fast_dim = 8
    slow_dim = 8

    t_fast_idx = torch.arange(t_fast).repeat_interleave(spatial_tokens)
    s_fast_idx = torch.arange(spatial_tokens).repeat(t_fast)
    x_fast = torch.zeros(1, t_fast * spatial_tokens, fast_dim)
    x_fast[0, :, 0] = t_fast_idx
    x_fast[0, :, 1] = s_fast_idx

    t_slow_idx = torch.arange(t_slow).repeat_interleave(spatial_tokens)
    s_slow_idx = torch.arange(spatial_tokens).repeat(t_slow)
    x_slow = torch.zeros(1, t_slow * spatial_tokens, slow_dim)
    x_slow[0, :, 0] = t_slow_idx
    x_slow[0, :, 1] = s_slow_idx

    y_fast = lift(x_slow, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    if y_fast.size(1) > x_fast.size(1):
        y_fast = y_fast[:, : x_fast.size(1)]
    y_fast = y_fast[:, :, : fast_dim // 2]

    mixer = MixedSplitAttention(
        embed_dim=fast_dim,
        num_heads=2,
        dropout=0.0,
        bias=True,
        qk_norm=False,
        split_dim=-1,
        split_sizes=(fast_dim // 2, fast_dim // 2),
        flex_attn=False,
    )
    _set_identity_mha(mixer.cross_branch.mha)

    t_acc, s_acc = _alignment(y_fast, spatial_tokens)
    print(f"lifted y_fast: t-major={t_acc:.3f}, s-major={s_acc:.3f}")

    out = mixer.cross_branch(x_fast[:, :, fast_dim // 2 :], y=y_fast, attn_mask=None, is_causal=False)
    t_acc, s_acc = _alignment(out, spatial_tokens)
    print(f"cross-branch out: t-major={t_acc:.3f}, s-major={s_acc:.3f}")


if __name__ == "__main__":
    main()
