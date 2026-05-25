"""
Probe prediction stability under random input perturbations.

Loads a checkpoint + its configs.yaml, pulls a few validation samples, perturbs
the input frames, and reports output sensitivity statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch.nn.attention as _torch_attn
import models.bcat as _bcat_mod
import models.multiscale_bcat as _ms_mod
from data_utils.collate import custom_collate
from dataset import get_dataset
from models.bcat import BCAT
from models.multiscale_bcat import MultiscaleBCAT
from symbol_utils.environment import SymbolicEnvironment

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

# Allow CPU execution by not forcing CUDNN-attention backend.
class _NoOpSdpaKernel:
    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


_torch_attn.sdpa_kernel = _NoOpSdpaKernel()
_bcat_mod.sdpa_kernel = _NoOpSdpaKernel()
_ms_mod.sdpa_kernel = _NoOpSdpaKernel()


@dataclass
class StabilityRow:
    eps: float
    input_rel_l2: float
    output_rel_l2: float
    output_max_abs: float
    gain_rel_l2: float


@dataclass
class ModelSpec:
    model_name: str
    model_cfg: OmegaConf
    x_num: int
    data_dim: int
    t_num: int
    input_len: int
    checkpoint_path: str
    full_cfg: OmegaConf


def _resolve_path(raw_path: str) -> str:
    p = Path(raw_path).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([Path.cwd() / p, SRC_ROOT / p, REPO_ROOT / p])
        if p.parts and p.parts[0] == "src":
            trimmed = Path(*p.parts[1:])
            candidates.extend([Path.cwd() / trimmed, SRC_ROOT / trimmed, REPO_ROOT / trimmed])

    for c in candidates:
        if c.is_file():
            return str(c)
    return raw_path


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


def _infer_model_name(model_cfg_name: str) -> str:
    if model_cfg_name == "bcat_auto":
        return "bcat"
    if model_cfg_name == "multiscale_bcat_auto":
        return "multiscale_bcat"
    raise ValueError(f"Unsupported model config name for this probe: {model_cfg_name}")


def _load_model_spec(args) -> ModelSpec:
    checkpoint_path = _resolve_path(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = Path(checkpoint_path).with_name("configs.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find configs.yaml next to checkpoint: {config_path}")

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    model_name = _infer_model_name(model_cfg.name)
    if args.model != "auto" and args.model != model_name:
        raise ValueError(f"--model={args.model} disagrees with checkpoint config model={model_name}")

    return ModelSpec(
        model_name=model_name,
        model_cfg=model_cfg,
        x_num=int(cfg.data.x_num),
        data_dim=int(cfg.data.max_output_dimension),
        t_num=int(cfg.data.t_num),
        input_len=int(cfg.input_len),
        checkpoint_path=checkpoint_path,
        full_cfg=cfg,
    )


def _make_model(spec: ModelSpec, device: torch.device, cpu_safe: bool) -> torch.nn.Module:
    model_cfg = OmegaConf.create(OmegaConf.to_container(spec.model_cfg, resolve=True))
    # This probe uses only `fwd`; disabling KV cache avoids eval-mode cache
    # plumbing in cache-specific encoder wrappers.
    model_cfg.kv_cache = False
    if device.type == "cpu" and cpu_safe:
        model_cfg.flex_attn = False

    if spec.model_name == "bcat":
        model = BCAT(model_cfg, spec.x_num, spec.data_dim, max_data_len=spec.t_num)
    elif spec.model_name == "multiscale_bcat":
        model = MultiscaleBCAT(model_cfg, spec.x_num, spec.data_dim, max_data_len=spec.t_num, eval_only=True)
    else:
        raise ValueError(f"Unknown model type: {spec.model_name}")
    return model.to(device)


def _to_device_batch(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device_batch(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device_batch(v, device) for v in x)
    return x


def _rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a64 = a.double()
    b64 = b.double()
    return ((a64 - b64).norm() / b64.norm().clamp_min(eps)).item()


def _parse_eps_list(raw: str) -> List[float]:
    vals = []
    for t in raw.split(","):
        t = t.strip()
        if t:
            vals.append(float(t))
    if not vals:
        raise ValueError("--eps must contain at least one value")
    return vals


def _default_times_like(data: torch.Tensor) -> torch.Tensor:
    bs, t_num = data.shape[:2]
    return torch.linspace(0.0, 10.0, t_num, device=data.device).view(1, t_num, 1).expand(bs, -1, -1).contiguous()


def _build_loader(spec: ModelSpec, max_samples: int):
    cfg = spec.full_cfg
    symbol_env = SymbolicEnvironment(cfg.symbol)
    datasets = get_dataset(cfg, symbol_env, split="val")
    type_order = list(cfg.data.types)
    chosen_type = type_order[0] if type_order and type_order[0] in datasets else next(iter(datasets.keys()))
    ds = datasets[chosen_type]

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            cfg.data.get("mixed_length", 0),
            cfg.input_len,
            cfg.symbol.pad_right,
        ),
    )
    return loader, chosen_type, max_samples


def _summarize(rows: Iterable[StabilityRow]) -> Dict[float, Dict[str, Tuple[float, float]]]:
    grouped: Dict[float, List[StabilityRow]] = {}
    for r in rows:
        grouped.setdefault(r.eps, []).append(r)

    out = {}
    for eps, items in grouped.items():
        stats = {}
        for field in ["input_rel_l2", "output_rel_l2", "output_max_abs", "gain_rel_l2"]:
            vals = torch.tensor([getattr(r, field) for r in items], dtype=torch.float64)
            stats[field] = (vals.mean().item(), vals.std(unbiased=False).item())
        out[eps] = stats
    return out


def _print_summary(summary: Dict[float, Dict[str, Tuple[float, float]]]) -> None:
    print("\neps        input_rel_l2        output_rel_l2       output_max_abs       gain_rel_l2")
    print("---------------------------------------------------------------------------------------")
    for eps in sorted(summary.keys()):
        s = summary[eps]
        print(
            f"{eps:8.2e}  "
            f"{s['input_rel_l2'][0]:12.5e}±{s['input_rel_l2'][1]:8.1e}  "
            f"{s['output_rel_l2'][0]:12.5e}±{s['output_rel_l2'][1]:8.1e}  "
            f"{s['output_max_abs'][0]:12.5e}±{s['output_max_abs'][1]:8.1e}  "
            f"{s['gain_rel_l2'][0]:12.5e}±{s['gain_rel_l2'][1]:8.1e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Probe output stability to random input perturbations.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["auto", "bcat", "multiscale_bcat"], default="auto")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eps", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--cpu-safe",
        type=int,
        default=1,
        help="When 1, disable flex_attn in model config on CPU to avoid block-mask device issues.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eps_list = _parse_eps_list(args.eps)
    if args.samples <= 0 or args.repeats <= 0:
        raise ValueError("--samples and --repeats must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    spec = _load_model_spec(args)
    model = _make_model(spec, device=device, cpu_safe=bool(args.cpu_safe))
    model.load_state_dict(_load_model_weights(spec.checkpoint_path), strict=True)
    model.eval()

    loader, ds_key, max_samples = _build_loader(spec, args.samples)

    print(f"Model: {spec.model_name}")
    print(f"Checkpoint: {spec.checkpoint_path}")
    print(f"Validation dataset key: {ds_key}")
    print(f"Device: {device}")
    print(f"Input_len={spec.input_len}, t_num={spec.t_num}, x_num={spec.x_num}, data_dim={spec.data_dim}")
    if device.type == "cpu" and bool(args.cpu_safe):
        print("CPU-safe mode active: flex_attn disabled.")

    rows: List[StabilityRow] = []
    processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device_batch(batch, device)
            data = batch["data"]
            times = batch.get("t", _default_times_like(data))

            base_output = model("fwd", data=data, times=times, input_len=spec.input_len).float()

            active_channels = None
            if "data_mask" in batch:
                active_channels = batch["data_mask"][0, 0, 0, 0, :].bool()

            base_in_slice = data[:, : spec.input_len].float()
            if active_channels is not None:
                base_in_slice = base_in_slice[..., active_channels]

            data_scale = base_in_slice.std().item()
            if data_scale == 0:
                data_scale = 1.0

            for eps in eps_list:
                for _ in range(args.repeats):
                    noise = torch.randn_like(data[:, : spec.input_len]) * (eps * data_scale)
                    if active_channels is not None:
                        mask = active_channels.view(1, 1, 1, 1, -1)
                        noise = noise * mask

                    perturbed = data.clone()
                    perturbed[:, : spec.input_len] = perturbed[:, : spec.input_len] + noise

                    out_pert = model("fwd", data=perturbed, times=times, input_len=spec.input_len).float()

                    in_slice = perturbed[:, : spec.input_len].float()
                    if active_channels is not None:
                        in_slice = in_slice[..., active_channels]

                    input_rel = _rel_l2(in_slice, base_in_slice)
                    output_rel = _rel_l2(out_pert, base_output)
                    output_max_abs = (out_pert - base_output).abs().max().item()
                    gain = output_rel / max(input_rel, 1e-12)

                    rows.append(
                        StabilityRow(
                            eps=eps,
                            input_rel_l2=input_rel,
                            output_rel_l2=output_rel,
                            output_max_abs=output_max_abs,
                            gain_rel_l2=gain,
                        )
                    )

            processed += 1
            if processed >= max_samples:
                break

    print(f"Processed validation samples: {processed}")
    print(f"Trials per eps: {processed * args.repeats}")
    _print_summary(_summarize(rows))


if __name__ == "__main__":
    main()
