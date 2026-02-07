import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import PoolFFN, SplitEncoder, pad_slow  # noqa: E402


class IdentityEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


def _set_first_token_pool_weights(pool_ffn: PoolFFN) -> None:
    in_dim = pool_ffn.in_dim
    hidden_dim = pool_ffn.fc1.out_features
    pool_ffn.fc1.weight.data.zero_()
    pool_ffn.fc1.bias.data.zero_()
    pool_ffn.fc2.weight.data.zero_()
    pool_ffn.fc2.bias.data.zero_()
    # Map the first token (first in_dim entries) through as identity.
    for i in range(min(in_dim, hidden_dim)):
        pool_ffn.fc1.weight.data[i, i] = 1.0
        pool_ffn.fc2.weight.data[i, i] = 1.0


def _expected_slow(x_fast: torch.Tensor, rate: int, spatial_tokens: int) -> torch.Tensor:
    x_pad, _ = pad_slow(x_fast, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    t_len = x_pad.size(1) // spatial_tokens
    num_blocks = t_len // rate
    x_ts = x_pad.view(1, t_len, spatial_tokens, -1)

    out = torch.zeros(1, num_blocks, spatial_tokens, x_fast.size(-1))
    for k in range(1, num_blocks):
        t_idx = (k - 1) * rate
        out[:, k] = x_ts[:, t_idx]
    return out.view(1, num_blocks * spatial_tokens, -1)


def main() -> None:
    torch.manual_seed(0)
    rate = 4
    spatial_tokens = 9
    time_len = 5
    dim = 8

    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    x_fast = torch.stack([t, s] + [t + s] * (dim - 2), dim=-1).float().unsqueeze(0)

    pool_ffn = PoolFFN(in_dim=dim, out_dim=dim, hidden_dim=dim, rate=rate, act="relu", dropout=0.0)
    _set_first_token_pool_weights(pool_ffn)

    encoder = SplitEncoder(
        encoder=IdentityEncoder(),
        rate=rate,
        pool_ffn=pool_ffn,
        time_dim=1,
        spatial_tokens=spatial_tokens,
    )

    _, slow = encoder(x_fast)
    expected = _expected_slow(x_fast, rate=rate, spatial_tokens=spatial_tokens)
    assert torch.equal(slow, expected)

    print("SplitEncoder pooling order check passed.")


if __name__ == "__main__":
    main()
