from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.multiscale_bcat import (
    build_fast_to_slow_mask,
    build_self_attn_mask,
    build_slow_to_fast_mask,
)


def _allowed_to_binary(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        blocked = mask
        return (~blocked).to(torch.float32)
    if torch.is_floating_point(mask):
        blocked = torch.isneginf(mask) | (mask <= -1e9)
        return (~blocked).to(torch.float32)
    return (mask != 0).to(torch.float32)


def _block_to_binary(mask) -> torch.Tensor:
    if hasattr(mask, "mask_mod") and hasattr(mask, "seq_lengths"):
        q_len, kv_len = mask.seq_lengths
        q_idx = torch.arange(q_len)
        kv_idx = torch.arange(kv_len)
        allow = mask.mask_mod(None, None, q_idx[:, None], kv_idx[None, :])
        return allow.to(torch.float32)
    dense = mask.to_dense()
    if dense.dim() > 2:
        dense = dense.reshape(-1, dense.shape[-2], dense.shape[-1])
        if dense.size(0) == 1:
            dense = dense[0]
        else:
            raise ValueError(f"Unexpected block mask batch shape: {dense.shape}")
    if dense.dim() == 0:
        dense = dense.reshape(1, 1)
    return _allowed_to_binary(dense)


def _report_diff(name: str, dense_bin: torch.Tensor, block_bin: torch.Tensor) -> None:
    if dense_bin.shape != block_bin.shape:
        print(f"{name}: shape mismatch dense={dense_bin.shape} block={block_bin.shape}")
        return
    mismatches = (dense_bin != block_bin).sum().item()
    total = dense_bin.numel()
    print(f"{name}: mismatches={mismatches}/{total}")


def main() -> None:
    patch_num = 4
    spatial_tokens = patch_num * patch_num
    time_len_fast = 8
    temporal_rate = 2

    if time_len_fast % temporal_rate != 0:
        raise ValueError("time_len_fast must be divisible by temporal_rate.")

    slow_time = (time_len_fast - 1) // temporal_rate + 1
    fast_self = build_self_attn_mask(
        time_len_fast, spatial_tokens, device=torch.device("cpu"), dtype=torch.float32
    )
    slow_self = build_self_attn_mask(
        slow_time, spatial_tokens, device=torch.device("cpu"), dtype=torch.float32
    )
    fast_to_slow = build_fast_to_slow_mask(
        time_len_fast,
        slow_time,
        temporal_rate,
        spatial_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    slow_to_fast = build_slow_to_fast_mask(
        time_len_fast,
        slow_time,
        temporal_rate,
        spatial_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    fast_self_block = build_self_attn_mask(
        time_len_fast, spatial_tokens, device=torch.device("cpu"), use_block_mask=True
    )
    slow_self_block = build_self_attn_mask(
        slow_time, spatial_tokens, device=torch.device("cpu"), use_block_mask=True
    )
    fast_to_slow_block = build_fast_to_slow_mask(
        time_len_fast,
        slow_time,
        temporal_rate,
        spatial_tokens,
        device=torch.device("cpu"),
        use_block_mask=True,
    )
    slow_to_fast_block = build_slow_to_fast_mask(
        time_len_fast,
        slow_time,
        temporal_rate,
        spatial_tokens,
        device=torch.device("cpu"),
        use_block_mask=True,
    )

    fast_self_binary = _allowed_to_binary(fast_self)
    slow_self_binary = _allowed_to_binary(slow_self)
    fast_to_slow_binary = _allowed_to_binary(fast_to_slow)
    slow_to_fast_binary = _allowed_to_binary(slow_to_fast)
    fast_self_block_binary = _block_to_binary(fast_self_block)
    slow_self_block_binary = _block_to_binary(slow_self_block)
    fast_to_slow_block_binary = _block_to_binary(fast_to_slow_block)
    slow_to_fast_block_binary = _block_to_binary(slow_to_fast_block)
    fast_self_diff = (fast_self_binary != fast_self_block_binary).to(torch.float32)
    slow_self_diff = (slow_self_binary != slow_self_block_binary).to(torch.float32)
    fast_to_slow_diff = (fast_to_slow_binary != fast_to_slow_block_binary).to(torch.float32)
    slow_to_fast_diff = (slow_to_fast_binary != slow_to_fast_block_binary).to(torch.float32)

    _report_diff("fast_self", fast_self_binary, fast_self_block_binary)
    _report_diff("slow_self", slow_self_binary, slow_self_block_binary)
    _report_diff("fast_to_slow", fast_to_slow_binary, fast_to_slow_block_binary)
    _report_diff("slow_to_fast", slow_to_fast_binary, slow_to_fast_block_binary)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_self_path = output_dir / "attn_mask_fast_self.png"
    slow_self_path = output_dir / "attn_mask_slow_self.png"
    fast_to_slow_path = output_dir / "attn_mask_fast_to_slow.png"
    slow_to_fast_path = output_dir / "attn_mask_slow_to_fast.png"
    fast_self_block_path = output_dir / "block_mask_fast_self.png"
    slow_self_block_path = output_dir / "block_mask_slow_self.png"
    fast_to_slow_block_path = output_dir / "block_mask_fast_to_slow.png"
    slow_to_fast_block_path = output_dir / "block_mask_slow_to_fast.png"
    fast_self_diff_path = output_dir / "mask_diff_fast_self.png"
    slow_self_diff_path = output_dir / "mask_diff_slow_self.png"
    fast_to_slow_diff_path = output_dir / "mask_diff_fast_to_slow.png"
    slow_to_fast_diff_path = output_dir / "mask_diff_slow_to_fast.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_self_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Fast self attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_self_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_self_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Slow self attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_self_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_to_slow_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Fast->Slow attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_to_slow_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_to_fast_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Slow->Fast attn_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_to_fast_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_self_block_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Fast self block_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_self_block_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_self_block_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Slow self block_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_self_block_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_to_slow_block_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Fast->Slow block_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_to_slow_block_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_to_fast_block_binary, cmap="gray", interpolation="nearest")
    ax.set_title("Slow->Fast block_mask (allowed=1)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_to_fast_block_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_self_diff, cmap="gray", interpolation="nearest")
    ax.set_title("Diff: fast self (1 = mismatch)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_self_diff_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_self_diff, cmap="gray", interpolation="nearest")
    ax.set_title("Diff: slow self (1 = mismatch)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_self_diff_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fast_to_slow_diff, cmap="gray", interpolation="nearest")
    ax.set_title("Diff: fast->slow (1 = mismatch)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(fast_to_slow_diff_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slow_to_fast_diff, cmap="gray", interpolation="nearest")
    ax.set_title("Diff: slow->fast (1 = mismatch)")
    ax.set_xlabel("key index")
    ax.set_ylabel("query index")
    fig.tight_layout()
    fig.savefig(slow_to_fast_diff_path, dpi=150)
    plt.close(fig)

    print(f"Saved {fast_self_path}")
    print(f"Saved {slow_self_path}")
    print(f"Saved {fast_to_slow_path}")
    print(f"Saved {slow_to_fast_path}")
    print(f"Saved {fast_self_block_path}")
    print(f"Saved {slow_self_block_path}")
    print(f"Saved {fast_to_slow_block_path}")
    print(f"Saved {slow_to_fast_block_path}")
    print(f"Saved {fast_self_diff_path}")
    print(f"Saved {slow_self_diff_path}")
    print(f"Saved {fast_to_slow_diff_path}")
    print(f"Saved {slow_to_fast_diff_path}")


if __name__ == "__main__":
    main()
