"""
Probe numerical drift between bf16 autocast and fp32 for BCAT variants.

Usage:
  python src/models/tests/test_mixed_precision_probe.py --model both --trials 3
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch.nn.attention as _torch_attn
import models.bcat as _bcat_mod
import models.multiscale_bcat as _ms_mod
from models.bcat import BCAT
from models.multiscale_bcat import MultiscaleBCAT

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def _no_op_sdpa_kernel(*args, **kwargs):
    yield


_torch_attn.sdpa_kernel = _no_op_sdpa_kernel
_bcat_mod.sdpa_kernel = _no_op_sdpa_kernel
_ms_mod.sdpa_kernel = _no_op_sdpa_kernel


@dataclass
class ProbeMetrics:
    output_rel_l2: float
    output_max_abs: float
    loss_rel: float
    grad_rel_l2: float
    grad_max_abs: float
    grad_cosine: float


@dataclass
class ModelSpec:
    model_name: str
    model_cfg: OmegaConf
    x_num: int
    data_dim: int
    t_num: int
    input_len: int
    checkpoint_path: str


def _resolve_checkpoint_path(model_name: str, args) -> str:
    if model_name == "bcat":
        raw = args.bcat_checkpoint
    elif model_name == "multiscale_bcat":
        raw = args.multiscale_checkpoint
    else:
        raise ValueError(f"Unknown model: {model_name}")

    p = Path(raw).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                Path.cwd() / p,
                SRC_ROOT / p,
                REPO_ROOT / p,
            ]
        )
        if p.parts and p.parts[0] == "src":
            trimmed = Path(*p.parts[1:])
            candidates.extend(
                [
                    Path.cwd() / trimmed,
                    SRC_ROOT / trimmed,
                    REPO_ROOT / trimmed,
                ]
            )

    for c in candidates:
        if c.is_file():
            return str(c)

    # keep original for clearer user-facing error
    return raw


def _load_model_spec(model_name: str, args) -> ModelSpec:
    checkpoint_path = _resolve_checkpoint_path(model_name, args)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = Path(checkpoint_path).with_name("configs.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find configs.yaml next to checkpoint: {config_path}")

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    data_cfg = cfg.data

    if model_name == "bcat" and model_cfg.name != "bcat_auto":
        raise ValueError(f"Expected bcat_auto model config, got: {model_cfg.name}")
    if model_name == "multiscale_bcat" and model_cfg.name != "multiscale_bcat_auto":
        raise ValueError(f"Expected multiscale_bcat_auto model config, got: {model_cfg.name}")

    return ModelSpec(
        model_name=model_name,
        model_cfg=model_cfg,
        x_num=int(data_cfg.x_num),
        data_dim=int(data_cfg.max_output_dimension),
        t_num=int(data_cfg.t_num),
        input_len=int(cfg.input_len),
        checkpoint_path=checkpoint_path,
    )


def _make_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    if spec.model_name == "bcat":
        return BCAT(spec.model_cfg, spec.x_num, spec.data_dim, max_data_len=spec.t_num).to(device)
    if spec.model_name == "multiscale_bcat":
        return MultiscaleBCAT(spec.model_cfg, spec.x_num, spec.data_dim, max_data_len=spec.t_num, eval_only=True).to(
            device
        )
    raise ValueError(f"Unknown model: {spec.model_name}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        out[k] = v
    return out


def _load_model_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return _normalize_state_dict_keys(payload["model"])
    if isinstance(payload, dict):
        return _normalize_state_dict_keys(payload)
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def _flatten_grads(named_params: Iterable[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    chunks = []
    for _, p in named_params:
        if p.grad is None:
            continue
        chunks.append(p.grad.detach().float().reshape(-1))
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks, dim=0)


def _rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a64 = a.double()
    b64 = b.double()
    num = (a64 - b64).norm(p=2)
    den = b64.norm(p=2).clamp_min(eps)
    return (num / den).item()


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a64 = a.double()
    b64 = b.double()
    den = (a64.norm(p=2) * b64.norm(p=2)).clamp_min(eps)
    cos = (a64.dot(b64) / den).item()
    if cos > 1.0:
        return 1.0
    if cos < -1.0:
        return -1.0
    return cos


def _single_pass(
    model: torch.nn.Module,
    data: torch.Tensor,
    times: torch.Tensor,
    target: torch.Tensor,
    input_len: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
        output = model("fwd", data=data, times=times, input_len=input_len)
        loss = F.mse_loss(output, target)
    loss.backward()
    grads = _flatten_grads(model.named_parameters())
    return output.detach().float(), loss.detach().float(), grads


def _probe_once(spec: ModelSpec, weights: Dict[str, torch.Tensor], args, device: torch.device, seed: int) -> ProbeMetrics:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_ref = _make_model(spec, device)
    model_amp = _make_model(spec, device)
    model_ref.load_state_dict(weights, strict=True)
    model_amp.load_state_dict(weights, strict=True)

    data = torch.randn(args.batch_size, spec.t_num, spec.x_num, spec.x_num, spec.data_dim, device=device)
    times = (
        torch.linspace(0.0, 1.0, spec.t_num, device=device)
        .view(1, spec.t_num, 1)
        .expand(args.batch_size, -1, -1)
        .contiguous()
    )
    out_len = spec.t_num - spec.input_len
    target = torch.randn(args.batch_size, out_len, spec.x_num, spec.x_num, spec.data_dim, device=device)

    out_ref, loss_ref, grad_ref = _single_pass(
        model=model_ref,
        data=data,
        times=times,
        target=target,
        input_len=spec.input_len,
        use_amp=False,
        amp_dtype=torch.bfloat16,
        device_type=device.type,
    )
    out_amp, loss_amp, grad_amp = _single_pass(
        model=model_amp,
        data=data,
        times=times,
        target=target,
        input_len=spec.input_len,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        device_type=device.type,
    )

    output_diff = (out_amp - out_ref).abs()
    grad_diff = (grad_amp - grad_ref).abs()

    if grad_ref.numel() == 0 or grad_amp.numel() == 0:
        grad_cos = float("nan")
        grad_rel = float("nan")
        grad_max = float("nan")
    else:
        grad_cos = _safe_cosine(grad_amp, grad_ref)
        grad_rel = _rel_l2(grad_amp, grad_ref)
        grad_max = grad_diff.max().item()

    loss_rel = abs(loss_amp.item() - loss_ref.item()) / max(abs(loss_ref.item()), 1e-12)

    return ProbeMetrics(
        output_rel_l2=_rel_l2(out_amp, out_ref),
        output_max_abs=output_diff.max().item(),
        loss_rel=loss_rel,
        grad_rel_l2=grad_rel,
        grad_max_abs=grad_max,
        grad_cosine=grad_cos,
    )


def _summarize(metrics: Iterable[ProbeMetrics]) -> Dict[str, Tuple[float, float]]:
    keys = ProbeMetrics.__dataclass_fields__.keys()
    vals = {k: [] for k in keys}
    for m in metrics:
        for k in keys:
            vals[k].append(getattr(m, k))

    out = {}
    for k, series in vals.items():
        tensor = torch.tensor(series, dtype=torch.float64)
        out[k] = (tensor.mean().item(), tensor.std(unbiased=False).item())
    return out


def _print_summary(model_name: str, summary: Dict[str, Tuple[float, float]]) -> None:
    print(f"\n=== {model_name} ===")
    print("metric                          mean             std")
    print("-" * 58)
    for key, (mean, std) in summary.items():
        print(f"{key:28s} {mean:12.5e}  {std:12.5e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe bf16 autocast drift against fp32.")
    parser.add_argument("--model", choices=["bcat", "multiscale_bcat", "both"], default="both")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Alias for --trials (kept for experiment-style wording).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--cpu-safe",
        type=int,
        default=1,
        help="When 1, disable flex_attn in loaded config to avoid CPU/CUDA block-mask mismatches.",
    )
    parser.add_argument(
        "--bcat-checkpoint",
        type=str,
        default="checkpoint/bcat/bcat_baseline_3/best-_l2_error.pth",
        help="Checkpoint to use for BCAT probe.",
    )
    parser.add_argument(
        "--multiscale-checkpoint",
        type=str,
        default="checkpoint/bcat/multiscale_bcat_33/best-_l2_error.pth",
        help="Checkpoint to use for multiscale BCAT probe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trials = args.steps if args.steps is not None else args.trials
    if trials <= 0:
        raise ValueError("Number of trials/steps must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model_names = ["bcat", "multiscale_bcat"] if args.model == "both" else [args.model]

    print(f"Device: {device}")
    print("Comparison: bf16 autocast vs fp32 (same checkpoint weights, same input batch)")
    print(f"Trials per model: {trials}")
    if device.type == "cpu" and bool(args.cpu_safe):
        print("CPU-safe mode: flex_attn disabled in model config for this probe run.")

    for model_name in model_names:
        spec = _load_model_spec(model_name, args)
        if device.type == "cpu" and bool(args.cpu_safe):
            # CPU-safe path: disable flex attention block masks, which can be
            # materialized on CUDA by backend internals and trigger
            # mixed-device errors.
            spec.model_cfg.flex_attn = False
        weights = _load_model_weights(spec.checkpoint_path)
        all_metrics = []
        for i in range(trials):
            run_seed = args.seed + i
            all_metrics.append(_probe_once(spec=spec, weights=weights, args=args, device=device, seed=run_seed))
        print(f"\nUsing checkpoint: {spec.checkpoint_path}")
        print(
            f"Shape from config: t_num={spec.t_num}, input_len={spec.input_len}, "
            f"x_num={spec.x_num}, data_dim={spec.data_dim}"
        )
        summary = _summarize(all_metrics)
        _print_summary(model_name, summary)


if __name__ == "__main__":
    main()
