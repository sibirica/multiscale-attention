import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import _downsample_mask, pad_slow  # noqa: E402


def _make_block_mask(time_len: int, spatial_tokens: int, rate: int) -> torch.Tensor:
    total = time_len * spatial_tokens
    mask = torch.zeros(total, total, dtype=torch.float32)
    for i in range(total):
        t_i = i // spatial_tokens
        for j in range(total):
            t_j = j // spatial_tokens
            if t_i // rate < t_j // rate:
                mask[i, j] = float("-inf")
    return mask


def main() -> None:
    rate = 4
    spatial_tokens = 9
    time_len = 5

    mask = _make_block_mask(time_len, spatial_tokens, rate)
    slow = _downsample_mask(mask, rate=rate, spatial_tokens=spatial_tokens, reduce_op="all", float_mode="all_masked")

    padded, pad_steps = pad_slow(mask, rate=rate, time_dim=-1, spatial_tokens=spatial_tokens, pad_value=float("-inf"))
    padded, _ = pad_slow(padded, rate=rate, time_dim=-2, spatial_tokens=spatial_tokens, pad_value=float("-inf"))
    expected_time = (time_len + pad_steps + rate - 1) // rate
    expected_len = expected_time * spatial_tokens
    assert slow.shape[-1] == expected_len and slow.shape[-2] == expected_len

    # Ensure no unmasked future block leaks after downsampling.
    slow_dense = slow
    for i in range(expected_len):
        t_i = i // spatial_tokens
        for j in range(expected_len):
            t_j = j // spatial_tokens
            if t_i < t_j and slow_dense[i, j] != float("-inf"):
                raise AssertionError("Downsampled mask allows future attention.")

    print("Mask downsample alignment check passed.")


if __name__ == "__main__":
    main()
