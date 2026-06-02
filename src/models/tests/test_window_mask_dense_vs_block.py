"""
Dense float masks vs flex BlockMask callback parity for sliding-window masks.

If these diverge, training paths can disagree with ``flex_attn=1``.

Run from the repository ``src`` directory (stdlib ``unittest`` only, no pytest):

    python models/tests/test_window_mask_dense_vs_block.py

or, from ``src``:

    python -m unittest discover -s models/tests -p test_window_mask_dense_vs_block.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import torch
except ImportError as exc:
    raise SystemExit("test_window_mask_dense_vs_block requires PyTorch; install torch to run this test.") from exc

from models.multiscale_bcat import (
    _block_fast_to_slow,
    _block_self_window,
    _block_slow_to_fast,
    build_fast_to_slow_mask,
    build_self_attn_mask,
    build_slow_to_fast_mask,
)


class TestWindowMaskDenseVsBlock(unittest.TestCase):
    def test_fast_to_slow_dense_matches_block_callbacks(self):
        device = torch.device("cpu")

        cases = [
            (2, 5, max(1, 5 // 2), 1),
            (2, 5, max(1, 5 // 2), 2),
            (3, 7, max(1, 7 // 3), 1),
            (5, 10, max(1, 10 // 5), 4),
        ]
        windows = [None, 2, 100]

        for rate, ft, st, spat in cases:
            for window in windows:
                dense = build_fast_to_slow_mask(
                    ft,
                    st,
                    rate,
                    spat,
                    device,
                    dtype=torch.float32,
                    use_block_mask=False,
                    window=window,
                )
                da = torch.isfinite(dense)
                Lq, Lk = dense.shape

                for qi in range(Lq):
                    for kj in range(Lk):
                        q = torch.tensor(qi, device=device)
                        kk = torch.tensor(kj, device=device)
                        bval = bool(_block_fast_to_slow(0, 0, q, kk, spat, rate, window=window).reshape(()).item())

                        self.assertEqual(
                            da[qi, kj].item(),
                            bval,
                            f"qi,kj={qi},{kj} ft={ft} st={st} rate={rate} S={spat} w={window}",
                        )

    def test_slow_to_fast_dense_matches_block_callbacks(self):
        device = torch.device("cpu")

        cases = [
            (2, 6, max(1, 6 // 2), 1),
            (3, 9, max(1, 9 // 3), 2),
        ]
        windows = [None, 2]

        for rate, ft, st, spat in cases:
            for window in windows:
                dense = build_slow_to_fast_mask(
                    ft,
                    st,
                    rate,
                    spat,
                    device,
                    dtype=torch.float32,
                    use_block_mask=False,
                    window=window,
                )
                da = torch.isfinite(dense)
                Lq, Lk = dense.shape

                for qi in range(Lq):
                    for kj in range(Lk):
                        q = torch.tensor(qi, device=device)
                        kk = torch.tensor(kj, device=device)
                        bval = bool(_block_slow_to_fast(0, 0, q, kk, spat, rate, window=window).reshape(()).item())

                        self.assertEqual(
                            da[qi, kj].item(),
                            bval,
                            f"qi,kj={qi},{kj} ft={ft} st={st} rate={rate} S={spat} w={window}",
                        )

    def test_self_window_dense_matches_block_callback(self):
        device = torch.device("cpu")

        for spat in [1, 2]:
            for tl in [3, 6]:
                for window in [1, 2, 100]:
                    dense = build_self_attn_mask(
                        tl,
                        spat,
                        device,
                        dtype=torch.float32,
                        use_block_mask=False,
                        window=window,
                    )
                    da = torch.isfinite(dense)
                    L = dense.shape[0]

                    for qi in range(L):
                        for kj in range(L):
                            q = torch.tensor(qi, device=device)
                            kk = torch.tensor(kj, device=device)
                            bval = bool(_block_self_window(0, 0, q, kk, spat, window).reshape(()).item())

                            self.assertEqual(
                                da[qi, kj].item(),
                                bval,
                                f"qi,kj={qi},{kj} tl={tl} S={spat} w={window}",
                            )


if __name__ == "__main__":
    unittest.main()
