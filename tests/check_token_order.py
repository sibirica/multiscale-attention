import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import lift, pad_slow  # noqa: E402


def _make_tokens(time_len: int, spatial_tokens: int) -> torch.Tensor:
    """
    Create tokens with channel-0=time index and channel-1=spatial index.
    Flattened order is time-major: (t0,s0..sS-1, t1,s0..sS-1, ...).
    """
    total = time_len * spatial_tokens
    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    return torch.stack([t, s], dim=-1).float().unsqueeze(0)


def _check_lift_order(time_len_fast: int, rate: int, spatial_tokens: int) -> None:
    slow_time_len = (time_len_fast + rate - 1) // rate
    z_slow = _make_tokens(slow_time_len, spatial_tokens)
    z_fast = lift(z_slow, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)

    expected_len = slow_time_len * rate * spatial_tokens
    assert z_fast.size(1) == expected_len

    # Trim to the original fast length if slow was padded.
    fast_len = time_len_fast * spatial_tokens
    z_fast = z_fast[:, :fast_len]

    t_fast = torch.arange(time_len_fast).repeat_interleave(spatial_tokens)
    s_fast = torch.arange(spatial_tokens).repeat(time_len_fast)
    t_slow = t_fast // rate

    expected = torch.stack([t_slow, s_fast], dim=-1).float().unsqueeze(0)
    assert torch.equal(z_fast, expected)


def _check_pad_slow(time_len_fast: int, rate: int, spatial_tokens: int) -> None:
    x_fast = _make_tokens(time_len_fast, spatial_tokens)
    padded, pad_steps = pad_slow(x_fast, rate=rate, time_dim=1, spatial_tokens=spatial_tokens)
    assert padded.size(1) == (time_len_fast + pad_steps) * spatial_tokens
    if pad_steps > 0:
        tail = padded[:, -pad_steps * spatial_tokens :]
        assert torch.all(tail == 0)


def main() -> None:
    torch.manual_seed(0)
    spatial_tokens = 9
    rate = 4
    for time_len_fast in (5, 8, 9, 16):
        _check_lift_order(time_len_fast, rate=rate, spatial_tokens=spatial_tokens)
        _check_pad_slow(time_len_fast, rate=rate, spatial_tokens=spatial_tokens)
    print("Token order checks passed.")


if __name__ == "__main__":
    main()
