import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.embedder import get_embedder  # noqa: E402


def _get_time_embeddings(embedder, times: torch.Tensor) -> torch.Tensor:
    if getattr(embedder, "time_embed_type", None) == "continuous":
        return embedder.time_proj(times)[:, :, None]
    return embedder.time_embeddings[:, : times.size(1)]


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    embedder = get_embedder(model_cfg.embedder, x_num, max_output_dim)

    batch_size = 1
    t_len = 3
    spatial_tokens = model_cfg.embedder.patch_num ** 2
    data = torch.zeros(batch_size, t_len, x_num, x_num, max_output_dim)
    times = torch.arange(t_len, dtype=torch.float32).view(1, t_len, 1)

    tokens = embedder.encode(data, times)  # (b, t*s, d)
    tokens = tokens.view(batch_size, t_len, spatial_tokens, -1)

    time_emb = _get_time_embeddings(embedder, times)  # (b, t, 1, d)
    time_delta = time_emb[:, 1:] - time_emb[:, :1]
    token_delta = tokens[:, 1:] - tokens[:, :1]

    # Across spatial tokens, the temporal delta should be constant (time-major order).
    max_err = (token_delta - time_delta).abs().max().item()
    if max_err > 1e-5:
        raise AssertionError(f"Unexpected token order; max error={max_err:.3e}.")

    print("Model token order check passed.")


if __name__ == "__main__":
    main()
