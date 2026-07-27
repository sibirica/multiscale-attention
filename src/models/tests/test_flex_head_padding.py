"""
Tests for head-dim padding used by MultiheadFlexAttention.

flex_attention's Triton kernel requires the attention head dim to be a power of
2. MultiheadFlexAttention pads q/k/v with zeros up to the next power of 2 and
keeps the softmax scale on the true head dim. Padding with zeros is exact:
the extra dimensions contribute nothing to the qk dot products or to the
weighted-sum output.

We validate the padding equivalence against F.scaled_dot_product_attention,
which is backend-agnostic and runs on CPU (flex_attention's compiled Triton
kernel does not), so the property under test is checked independently of flex.

Usage (from the repository's ``src/`` directory):
  python models/tests/test_flex_head_padding.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

from models.attention_utils import MultiheadFlexAttention, _next_power_of_2


def test_next_power_of_2():
    cases = {1: 1, 2: 2, 3: 4, 7: 8, 8: 8, 32: 32, 56: 64, 64: 64, 65: 128, 96: 128}
    for n, expected in cases.items():
        assert _next_power_of_2(n) == expected, (n, _next_power_of_2(n), expected)


def test_module_padding_attributes():
    # embed_dim=448, n_head=8 -> head_dim=56 (not a power of 2)
    attn = MultiheadFlexAttention(embed_dim=448, num_heads=8, qk_norm=True)
    assert attn.head_dim == 56
    assert attn.padded_head_dim == 64
    assert abs(attn.attn_scale - 56**-0.5) < 1e-12

    # a power-of-2 head dim must not be padded
    attn2 = MultiheadFlexAttention(embed_dim=512, num_heads=8)
    assert attn2.head_dim == 64
    assert attn2.padded_head_dim == 64


def test_zero_padding_is_exact():
    """Zero-padding the head dim (with scale on the true dim) is exact."""
    torch.manual_seed(0)
    b, h, s = 2, 8, 20
    for head_dim in (56, 40, 96):  # non powers of 2
        q, k, v = (torch.randn(b, h, s, head_dim, dtype=torch.float64) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)  # default scale = head_dim**-0.5

        pad = _next_power_of_2(head_dim) - head_dim
        qp, kp, vp = (F.pad(x, (0, pad)) for x in (q, k, v))
        padded = F.scaled_dot_product_attention(qp, kp, vp, scale=head_dim**-0.5)[..., :head_dim]

        assert torch.allclose(padded, ref, atol=1e-12), (head_dim, (padded - ref).abs().max().item())

    # sanity: using the padded dim for the scale (the bug we avoid) is *not* equivalent
    head_dim = 56
    q, k, v = (torch.randn(b, h, s, head_dim, dtype=torch.float64) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    pad = _next_power_of_2(head_dim) - head_dim
    qp, kp, vp = (F.pad(x, (0, pad)) for x in (q, k, v))
    wrong = F.scaled_dot_product_attention(qp, kp, vp)[..., :head_dim]  # default scale uses padded dim
    assert not torch.allclose(wrong, ref, atol=1e-6)


if __name__ == "__main__":
    test_next_power_of_2()
    test_module_padding_attributes()
    test_zero_padding_is_exact()
    print("All head-dim padding tests passed.")
