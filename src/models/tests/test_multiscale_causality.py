"""
Causality unit tests for the multiscale BCAT.

Strategy:
  Build a tiny model with frozen random weights, run a deterministic forward
  pass, then perturb the input at a future fast time step and re-run. Any
  output position strictly before the perturbation must remain bitwise
  unchanged for a properly causal model.

Usage (from the repository's ``src/`` directory):
  python models/tests/test_multiscale_causality.py
"""

from __future__ import annotations

import os
import sys
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

# IMPORTANT: BCAT and MultiscaleBCAT both wrap their attention call in
# ``with sdpa_kernel(SDPBackend.CUDNN_ATTENTION)``, which forces a CUDNN/Flash
# backend that requires bf16 inputs. bf16's ~1e-2 quantization noise is large
# enough to mask the small-magnitude leakage we want to detect, so we patch
# ``sdpa_kernel`` to be a no-op for the duration of the test, letting the
# default backend selector pick the float32-capable MATH kernel.
import torch.nn.attention as _torch_attn


@contextmanager
def _no_op_sdpa_kernel(*args, **kwargs):
    yield


_torch_attn.sdpa_kernel = _no_op_sdpa_kernel
# Also patch the already-imported module-level reference inside bcat.py / multiscale_bcat.py.
import models.bcat as _bcat_mod
import models.multiscale_bcat as _ms_mod

_bcat_mod.sdpa_kernel = _no_op_sdpa_kernel
_ms_mod.sdpa_kernel = _no_op_sdpa_kernel

from models.bcat import BCAT
from models.multiscale_bcat import MultiscaleBCAT


# Small but non-trivial sizes; rate=2 matches the production setup. patch_num=2
# keeps spatial token count at 4, which is enough to verify that the bug is not
# a 1D special case. n_layer is set to the production depth so any leak that
# only manifests after several rounds of fast<->slow exchange is captured.
T_NUM = 14
INPUT_LEN = 3
PATCH_NUM = 2
X_NUM = 4
DATA_DIM = 2
DIM_EMB = 16
DIM_FFN = 32
N_HEAD = 2
N_LAYER = 6
RATE = 2


def _bcat_config():
    return OmegaConf.create(
        dict(
            name="bcat_auto",
            n_layer=N_LAYER,
            dim_emb=DIM_EMB,
            dim_ffn=DIM_FFN,
            dropout=0.0,
            attn_dropout=0.0,
            n_head=N_HEAD,
            norm_first=True,
            qk_norm=False,
            norm="layer",
            activation="gelu",
            rotary=False,
            flex_attn=False,
            kv_cache=False,
            patch_num=PATCH_NUM,
            patch_num_output=PATCH_NUM,
            embedder=dict(
                type="conv",
                dim=DIM_EMB,
                patch_num=PATCH_NUM,
                patch_num_output=PATCH_NUM,
                time_embed="learnable",
                max_time_len=T_NUM + 4,
                conv_dim=8,
                early_conv=False,
                deep=False,
            ),
        )
    )


def _multiscale_config():
    return OmegaConf.create(
        dict(
            name="multiscale_bcat_auto",
            n_layer=N_LAYER,
            dim_emb=DIM_EMB,
            slow_dim=DIM_EMB,
            dim_ffn=DIM_FFN,
            pool_dim=DIM_FFN,
            dropout=0.0,
            attn_dropout=0.0,
            n_head=N_HEAD,
            norm_first=True,
            qk_norm=False,
            norm="layer",
            activation="gelu",
            recombine_activation="gelu",
            rotary=False,
            flex_attn=False,
            kv_cache=False,
            rate=RATE,
            shared_scale_ffn=False,
            limit_window=True,
            self_window=4,
            fast_to_slow_window=2,
            slow_to_fast_window=RATE,
            patch_num=PATCH_NUM,
            patch_num_output=PATCH_NUM,
            embedder=dict(
                type="conv",
                dim=DIM_EMB,
                patch_num=PATCH_NUM,
                patch_num_output=PATCH_NUM,
                time_embed="learnable",
                max_time_len=T_NUM + 4,
                conv_dim=8,
                early_conv=False,
                deep=False,
            ),
        )
    )


def _set_disable_slow_scale(model: MultiscaleBCAT, value: bool) -> None:
    """Override the hardcoded flag in every encoder layer."""
    for layer in model.transformer.layers:
        layer.disable_slow_scale = value


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    bs = 2
    data = torch.randn(bs, T_NUM, X_NUM, X_NUM, DATA_DIM, device=DEVICE)
    times = (
        torch.linspace(0.0, 1.0, T_NUM, device=DEVICE)
        .view(1, T_NUM, 1)
        .expand(bs, -1, -1)
        .contiguous()
    )
    return data, times


@torch.no_grad()
def _run_fwd(model: torch.nn.Module, data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Forward pass in fp32 (no autocast) so that bf16 quantization noise does not mask the leak."""
    model.eval()
    return model("fwd", data=data, times=times, input_len=INPUT_LEN)


@torch.no_grad()
def _run_generate(
    model: torch.nn.Module,
    data_input: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    data_mask = torch.ones(1, 1, 1, 1, data_input.size(-1), device=data_input.device)
    return model(
        "generate",
        data_input=data_input,
        times=times,
        input_len=INPUT_LEN,
        data_mask=data_mask,
    )


def causality_check(
    label: str,
    model_factory: Callable[[], torch.nn.Module],
    t_perturb_fast: int,
    rtol: float = 0.0,
    atol: float = 1e-6,
):
    """Compare ``model.fwd(X)`` and ``model.fwd(X')`` where X' differs from X
    only at fast time ``t_perturb_fast``. All output positions strictly before
    ``t_perturb_fast - 1`` must be unchanged for a causal model. (Position
    ``t_perturb_fast - 1`` predicts position ``t_perturb_fast`` and is allowed
    to depend on the data at ``t_perturb_fast - 1``, but the perturbation is
    *at* ``t_perturb_fast``, so technically position ``t_perturb_fast - 1``
    *should* still be unchanged — see note below.)

    Note on indexing: ``BCAT.fwd`` and ``MultiscaleBCAT.fwd`` both drop the
    last time step (``data[:, :-1]``) before encoding, so a perturbation at
    fast index ``t_perturb_fast == T_NUM - 1`` is dropped entirely and would
    not test anything. We therefore restrict ``t_perturb_fast < T_NUM - 1``.
    The model output at fast position ``k`` (after the ``[:, input_len-1:]``
    slice) corresponds to the prediction conditioned on data through time
    ``input_len - 1 + k``. For causality, perturbing fast index
    ``t_perturb_fast`` may only affect output positions ``k`` such that
    ``input_len - 1 + k >= t_perturb_fast``, i.e. ``k >= t_perturb_fast -
    input_len + 1``.
    """

    assert 0 <= t_perturb_fast < T_NUM - 1
    model = model_factory()

    data, times = _make_inputs()
    data_perturbed = data.clone()
    data_perturbed[:, t_perturb_fast] += torch.randn_like(data_perturbed[:, t_perturb_fast])

    y = _run_fwd(model, data, times)
    y_p = _run_fwd(model, data_perturbed, times)

    # Output positions correspond to fast indices [INPUT_LEN-1, ..., T_NUM-2].
    # We expect them to be unchanged for fast indices < t_perturb_fast.
    # Index k (in y) -> fast index INPUT_LEN-1+k.
    diffs = (y - y_p).abs().flatten(2).max(dim=-1).values  # (bs, output_len)
    output_len = diffs.size(1)

    leak_count = 0
    print(f"\n===== {label} =====")
    print(
        f"perturbing fast index {t_perturb_fast} "
        f"(should only affect outputs predicting fast indices >= {t_perturb_fast})"
    )
    for k in range(output_len):
        fast_idx = INPUT_LEN - 1 + k
        max_diff = diffs[:, k].max().item()
        causal_ok = (fast_idx >= t_perturb_fast) or (max_diff <= atol)
        marker = "ok  " if causal_ok else "LEAK"
        if not causal_ok:
            leak_count += 1
        print(
            f"  output[{k:2d}] (fast idx {fast_idx:2d}): max abs diff = {max_diff:.3e} "
            f"[{'should match' if fast_idx < t_perturb_fast else 'allowed to differ'}] {marker}"
        )
    if leak_count == 0:
        print(f"  RESULT: causal (no leaks)")
    else:
        print(f"  RESULT: NOT CAUSAL ({leak_count} leak(s))")
    return leak_count


def fwd_vs_generate_check(
    label: str,
    model_factory: Callable[[], torch.nn.Module],
    atol: float = 1e-4,
):
    """Compare ``model.fwd(data)[:, 0]`` (first prediction under teacher
    forcing on the full ``T_NUM`` window) with ``model.generate(data[:input_len])[:, 0]``
    (first autoregressive step).

    For a causal model both predictions are conditioned on exactly
    ``data[:, :input_len]`` and must therefore be (numerically) identical.
    A small residual difference is unavoidable because ``fwd`` and ``generate``
    feed different sequence lengths into ``F.scaled_dot_product_attention``,
    which can pick different kernels or accumulation orders. Empirically this
    background noise is on the order of 1e-5 (BCAT baseline below).
    A genuine future-information leak through the slow stream produces
    differences orders of magnitude larger.
    """
    model = model_factory()
    data, times = _make_inputs()

    out_fwd = _run_fwd(model, data, times)  # (bs, output_len, x_num, x_num, dim)
    out_gen = _run_generate(model, data[:, :INPUT_LEN], times)  # (bs, output_len, ...)

    diff = (out_fwd[:, 0] - out_gen[:, 0]).abs().flatten(1).max(dim=-1).values
    max_diff = diff.max().item()
    consistent = max_diff <= atol
    print(f"\n===== fwd-vs-generate: {label} =====")
    print(
        f"  max |fwd[0] - gen[0]| over batch = {max_diff:.3e}; "
        f"tolerance = {atol:.0e}; "
        f"{'CONSISTENT' if consistent else 'INCONSISTENT (leak)'}"
    )
    return 0 if consistent else 1


@torch.no_grad()
def dense_vs_kv_rollout_generate_check(
    label: str,
    disable_slow_scale: bool,
    atol: float = 5e-3,
):
    """Two MultiscaleBCAT instances with identical weights: ``kv_cache=0`` vs ``kv_cache=1``
    dense autoregressive ``generate``. Outputs should match within SDPA/backend noise."""
    cfg_base = OmegaConf.create(OmegaConf.to_container(_multiscale_config(), resolve=True))
    cfg_base.kv_cache = False

    cfg_kv = OmegaConf.create(OmegaConf.to_container(_multiscale_config(), resolve=True))
    cfg_kv.kv_cache = True

    m_dense = MultiscaleBCAT(cfg_base, X_NUM, DATA_DIM, max_data_len=T_NUM).to(DEVICE)
    m_kv = MultiscaleBCAT(cfg_kv, X_NUM, DATA_DIM, max_data_len=T_NUM).to(DEVICE)
    m_kv.load_state_dict(m_dense.state_dict())

    dr = disable_slow_scale
    _set_disable_slow_scale(m_dense, dr)
    _set_disable_slow_scale(m_kv, dr)

    data, times = _make_inputs()
    data_mask = torch.ones(
        1, 1, 1, 1, data.size(-1), device=data.device, dtype=data.dtype
    )

    g_dense = m_dense(
        "generate",
        data_input=data[:, :INPUT_LEN],
        times=times,
        input_len=INPUT_LEN,
        data_mask=data_mask,
    )
    g_kv = m_kv(
        "generate",
        data_input=data[:, :INPUT_LEN],
        times=times,
        input_len=INPUT_LEN,
        data_mask=data_mask,
    )

    max_diff = (g_dense - g_kv).abs().max().item()
    ok = max_diff <= atol or torch.allclose(g_dense, g_kv, rtol=1e-3, atol=atol)
    suffix = "(disable_slow_scale=True)" if dr else "(disable_slow_scale=False)"
    print(f"\n===== dense vs kv rollout generate: {label} {suffix} =====")
    print(
        f"  max |dense - kv| = {max_diff:.3e}; atol={atol:.0e}; "
        f"{'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main():
    torch.manual_seed(0)

    n_leaks = {}

    bcat_factory = lambda: BCAT(_bcat_config(), X_NUM, DATA_DIM, max_data_len=T_NUM).to(DEVICE)

    def make_ms_disabled():
        m = MultiscaleBCAT(_multiscale_config(), X_NUM, DATA_DIM, max_data_len=T_NUM).to(DEVICE)
        _set_disable_slow_scale(m, True)
        return m

    def make_ms_full():
        m = MultiscaleBCAT(_multiscale_config(), X_NUM, DATA_DIM, max_data_len=T_NUM).to(DEVICE)
        _set_disable_slow_scale(m, False)
        return m

    # Probe several perturbation positions, including the slow-to-fast
    # boundary t = (s+1)*rate - 1 (e.g., t = 1, 3, 5, ...) where slow_s and
    # fast_t can read each other. Also probe the off-boundary positions to
    # catch any leak that depends on alignment.
    perturb_positions = [t for t in range(2, T_NUM - 1)]

    for label, factory in [
        ("BCAT (baseline)", bcat_factory),
        ("MultiscaleBCAT (disable_slow_scale=True)", make_ms_disabled),
        ("MultiscaleBCAT (disable_slow_scale=False)", make_ms_full),
    ]:
        total = 0
        for t in perturb_positions:
            total += causality_check(label, factory, t_perturb_fast=t)
        n_leaks[label] = total

    # fwd-vs-generate consistency. For the first generated step, both
    # `fwd` and `generate` are conditioned on exactly ``data[:, :input_len]``,
    # so a causal model must produce identical predictions.
    n_leaks["fwd_gen_bcat"] = fwd_vs_generate_check("BCAT (baseline)", bcat_factory)
    n_leaks["fwd_gen_ms_disabled"] = fwd_vs_generate_check(
        "MultiscaleBCAT (disable_slow_scale=True)",
        make_ms_disabled,
    )
    n_leaks["fwd_gen_ms_full"] = fwd_vs_generate_check(
        "MultiscaleBCAT (disable_slow_scale=False)",
        make_ms_full,
    )

    n_leaks["dense_kv_ms_disabled"] = dense_vs_kv_rollout_generate_check(
        "MultiscaleBCAT",
        disable_slow_scale=True,
    )
    n_leaks["dense_kv_ms_full"] = dense_vs_kv_rollout_generate_check(
        "MultiscaleBCAT",
        disable_slow_scale=False,
    )

    print()
    print("=" * 60)
    print(f"Summary: {n_leaks}")
    return n_leaks


if __name__ == "__main__":
    leaks = main()
