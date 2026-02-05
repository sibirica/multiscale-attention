import inspect
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import attention_utils  # noqa: E402
from models.build_model import build_model  # noqa: E402
from symbol_utils.environment import SymbolicEnvironment  # noqa: E402


def _summarize_value(value):
    if value is None:
        return "None"
    if torch.is_tensor(value):
        return f"Tensor{tuple(value.shape)}"
    return repr(value)


def _wrap_forward(forward_fn, records, name):
    sig = inspect.signature(forward_fn)

    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        args_map = bound.arguments
        record = {
            "module": name,
            "key_padding_mask": _summarize_value(args_map.get("key_padding_mask")),
            "attn_mask": _summarize_value(args_map.get("attn_mask")),
            "block_mask": _summarize_value(args_map.get("block_mask")),
            "is_causal": _summarize_value(args_map.get("is_causal")),
            "rotary_emb": _summarize_value(args_map.get("rotary_emb")),
        }
        records.append(record)
        return forward_fn(self, *args, **kwargs)

    return wrapper


def main():
    records = []
    attention_utils.MultiheadAttention.forward = _wrap_forward(
        attention_utils.MultiheadAttention.forward, records, "MultiheadAttention"
    )
    attention_utils.MultiheadFlexAttention.forward = _wrap_forward(
        attention_utils.MultiheadFlexAttention.forward, records, "MultiheadFlexAttention"
    )

    with initialize_config_dir(config_dir=str(ROOT / "src" / "configs"), version_base=None):
        cfg = compose(config_name="main")

    cfg.cpu = True
    cfg.compile = False
    cfg.use_wandb = False
    cfg.model.kv_cache = 0

    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    model = modules["model"]
    model.eval()

    bs = 1
    t_num = int(cfg.input_len) + 1
    x_num = int(cfg.data.x_num)
    data_dim = int(cfg.data.max_output_dimension)

    data = torch.randn(bs, t_num, x_num, x_num, data_dim)
    times = torch.arange(t_num, dtype=torch.float32).view(1, t_num, 1)

    encoded = model.embedder.encode(data, times)
    data_len = encoded.size(1)
    mask = model.mask[:data_len, :data_len]

    with sdpa_kernel(SDPBackend.MATH):
        _ = model.transformer(encoded, mask=mask)

    keys = ["key_padding_mask", "attn_mask", "block_mask", "is_causal", "rotary_emb"]
    non_null_counts = {k: sum(1 for r in records if r[k] != "None") for k in keys}

    print("total_calls", len(records))
    print("first_calls", records[:5])
    print("non_null_counts", non_null_counts)


if __name__ == "__main__":
    main()
