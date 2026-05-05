"""
Static visualization of multiscale attention masks.

Run from the repository's `src/` directory so that `models/...` imports resolve.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from models.multiscale_bcat import (
    build_self_attn_mask,
    build_fast_to_slow_mask,
    build_slow_to_fast_mask,
)


def _show(name: str, mask: torch.Tensor) -> None:
    allowed = (mask == 0).int()
    print(f"--- {name} (shape={tuple(mask.shape)}; 1=allowed, 0=blocked) ---")
    print(allowed.numpy())
    print()


def main():
    rate = 2
    fast_time = 8
    slow_time = (fast_time - 1) // rate + 1
    spatial_tokens = 1
    device = torch.device("cpu")

    print(f"rate={rate}, fast_time={fast_time}, slow_time={slow_time}, spatial_tokens={spatial_tokens}")
    print()

    fast_self = build_self_attn_mask(fast_time, spatial_tokens, device, dtype=torch.float32)
    slow_self = build_self_attn_mask(slow_time, spatial_tokens, device, dtype=torch.float32)
    f2s = build_fast_to_slow_mask(fast_time, slow_time, rate, spatial_tokens, device, dtype=torch.float32)
    s2f = build_slow_to_fast_mask(fast_time, slow_time, rate, spatial_tokens, device, dtype=torch.float32)

    _show("fast_self_attn_mask  (rows=fast queries, cols=fast keys)", fast_self)
    _show("slow_self_attn_mask  (rows=slow queries, cols=slow keys)", slow_self)
    _show("fast_to_slow_mask    (rows=fast queries, cols=slow keys)", f2s)
    _show("slow_to_fast_mask    (rows=slow queries, cols=fast keys)", s2f)

    # Quantify the leak: how many *future* fast keys can each slow query attend to?
    s2f_allowed = (s2f == 0)
    fast_t = torch.arange(fast_time)
    slow_t = torch.arange(slow_time)
    # "future" = fast_t > slow_t * rate - 1, i.e. fast_t >= slow_t * rate (above the diagonal of the causal version)
    causal_upper_bound_per_slow = (slow_t + 1) * rate - 1  # max fast index slow_s should be able to read
    print("Leak diagnosis (per slow time step):")
    print(f"  fast_time = {fast_time}, slow_time = {slow_time}")
    for s in range(slow_time):
        attended = fast_t[s2f_allowed[s]].tolist()
        max_allowed = causal_upper_bound_per_slow[s].item()
        future = [t for t in attended if t > max_allowed]
        print(
            f"  slow_{s}: attends fast_{attended}; "
            f"causal upper-bound is fast_{max_allowed}; "
            f"FUTURE leak = {future}"
        )


if __name__ == "__main__":
    main()
