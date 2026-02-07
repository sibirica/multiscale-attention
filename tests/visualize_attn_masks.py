from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.bcat import block_lower_triangular_mask
from models.multiscale_utils import _downsample_attn_mask


def _attn_to_binary(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        blocked = mask
    else:
        blocked = torch.isneginf(mask) | (mask <= -1e9)
    return (~blocked).to(torch.float32)


def main() -> None:
    patch_num = 4
    spatial_tokens = patch_num * patch_num
    time_len_fast = 8
    temporal_rate = 2

    if time_len_fast % temporal_rate != 0:
        raise ValueError("time_len_fast must be divisible by temporal_rate.")

    block_size = spatial_tokens
    block_num = time_len_fast
    fast_attn_mask = block_lower_triangular_mask(block_size, block_num, use_float=True)
    slow_attn_mask = _downsample_attn_mask(
        fast_attn_mask, rate=temporal_rate, spatial_tokens=spatial_tokens
    )

    fast_binary = _attn_to_binary(fast_attn_mask)
    slow_binary = _attn_to_binary(slow_attn_mask)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_path = output_dir / "attn_mask_fast.png"
    slow_path = output_dir / "attn_mask_slow.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Fast attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Slow attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_path, dpi=150)
    plt.close(fig)

    print(f"Saved {fast_path}")
    print(f"Saved {slow_path}")


if __name__ == "__main__":
    main()
