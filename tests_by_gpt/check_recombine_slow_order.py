import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_utils import RecombineDecoder, lift  # noqa: E402


def _make_slow_tokens(time_len: int, spatial_tokens: int, dim: int) -> torch.Tensor:
    total = time_len * spatial_tokens
    t = torch.arange(time_len).repeat_interleave(spatial_tokens)
    s = torch.arange(spatial_tokens).repeat(time_len)
    tokens = torch.zeros(1, total, dim)
    tokens[0, :, 0] = t
    tokens[0, :, 1] = s
    return tokens


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")

    spatial_tokens = model_cfg.embedder.patch_num ** 2
    rate = int(model_cfg.rate)
    fast_dim = int(model_cfg.dim_emb)
    slow_dim = int(model_cfg.get("slow_dim", model_cfg.dim_emb))

    t_fast = 5
    t_slow = (t_fast + rate - 1) // rate

    z_fast = torch.zeros(1, t_fast * spatial_tokens, fast_dim)
    z_slow = _make_slow_tokens(t_slow, spatial_tokens, slow_dim)

    decoder = RecombineDecoder(
        fast_embed_dim=fast_dim,
        slow_embed_dim=slow_dim,
        rate=rate,
        hidden_dim=int(model_cfg.dim_ffn),
        act=model_cfg.get("activation", "gelu"),
        dropout=0.0,
        time_dim=1,
        spatial_tokens=spatial_tokens,
    )

    z_slow_lift = lift(
        z_slow,
        rate=rate,
        time_dim=decoder.time_dim,
        spatial_tokens=decoder.spatial_tokens,
    )
    if z_slow_lift.size(decoder.time_dim) > z_fast.size(decoder.time_dim):
        z_slow_lift = z_slow_lift.narrow(decoder.time_dim, 0, z_fast.size(decoder.time_dim))

    t_expected = torch.arange(t_fast).repeat_interleave(spatial_tokens) // rate
    s_expected = torch.arange(spatial_tokens).repeat(t_fast)
    if not torch.equal(z_slow_lift[0, :, 0].cpu(), t_expected):
        raise AssertionError("Recombine slow lift: time-major ordering mismatch.")
    if not torch.equal(z_slow_lift[0, :, 1].cpu(), s_expected):
        raise AssertionError("Recombine slow lift: spatial ordering mismatch.")

    print("Recombine slow lift order check passed.")


if __name__ == "__main__":
    main()
