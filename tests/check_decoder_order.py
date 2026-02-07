import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.embedder import get_embedder  # noqa: E402


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    embedder = get_embedder(model_cfg.embedder, x_num, max_output_dim)

    batch_size = 1
    t_len = 3
    patch_num = model_cfg.embedder.patch_num_output
    spatial_tokens = patch_num ** 2
    dim = model_cfg.embedder.dim

    tokens = torch.zeros(batch_size, t_len * spatial_tokens, dim)
    spatial_index = torch.arange(spatial_tokens).repeat(t_len)
    time_index = torch.arange(t_len).repeat_interleave(spatial_tokens)
    tokens[0, :, 0] = spatial_index
    tokens[0, :, 1] = time_index

    rearrange_layer = embedder.post_proj[0]
    out = rearrange_layer(tokens)
    out = out.view(batch_size, t_len, dim, patch_num, patch_num)

    expected_spatial = torch.arange(spatial_tokens).view(patch_num, patch_num)
    for t in range(t_len):
        spatial_slice = out[0, t, 0]
        time_slice = out[0, t, 1]
        if not torch.equal(spatial_slice, expected_spatial):
            raise AssertionError("Decoder spatial ordering does not match (h,w) flattening.")
        if not torch.all(time_slice == t):
            raise AssertionError("Decoder time ordering does not match time-major flattening.")

    print("Decoder order check passed.")


if __name__ == "__main__":
    main()
