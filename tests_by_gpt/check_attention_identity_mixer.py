import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import MixedSplitAttention  # noqa: E402


def _set_identity_mha(mha: torch.nn.Module) -> None:
    embed_dim = mha.embed_dim
    eye = torch.eye(embed_dim)
    for layer in (mha.linear_q, mha.linear_k, mha.linear_v, mha.out_proj):
        layer.weight.data.copy_(eye)
        if layer.bias is not None:
            layer.bias.data.zero_()


def main() -> None:
    torch.manual_seed(0)
    embed_dim = 4
    num_heads = 2
    seq_len = 6

    mixer = MixedSplitAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        qk_norm=False,
        split_dim=-1,
        split_sizes=(2, 2),
        flex_attn=False,
    )

    _set_identity_mha(mixer.self_branch.mha)
    _set_identity_mha(mixer.cross_branch.mha)

    t = torch.arange(seq_len)
    s = torch.arange(seq_len)
    x = torch.stack([t, s, t, s], dim=-1).float().unsqueeze(0)
    y = x[:, :, 2:4]

    attn_mask = torch.full((seq_len, seq_len), float("-inf"))
    attn_mask.fill_diagonal_(0.0)

    out = mixer(x=x, y=y, attn_mask=attn_mask, is_causal=False)
    if not torch.equal(out, x):
        raise AssertionError("Identity attention mixer did not preserve token order.")

    print("Attention identity mixer check passed.")


if __name__ == "__main__":
    main()
