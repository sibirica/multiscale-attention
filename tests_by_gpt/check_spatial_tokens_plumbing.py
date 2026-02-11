import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_bcat import MultiscaleBCAT  # noqa: E402
from models.attention_utils import MultiheadAttention  # noqa: E402


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    model = MultiscaleBCAT(model_cfg, x_num=x_num, max_output_dim=max_output_dim)
    model.eval()

    recorded = {}
    expected = model_cfg.embedder.patch_num ** 2

    original_transformer_forward = model.transformer.forward

    def wrapped_forward(*args, **kwargs):
        recorded["spatial_tokens"] = kwargs.get("spatial_tokens")
        return original_transformer_forward(*args, **kwargs)

    model.transformer.forward = wrapped_forward

    data = torch.zeros(1, 3, x_num, x_num, max_output_dim)
    times = torch.arange(3, dtype=torch.float32).view(1, 3, 1)
    def _naive_attention_forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal=False,
        rotary_emb=None,
        cache=None,
    ):
        if key_padding_mask is not None or block_mask is not None or is_causal:
            raise RuntimeError("Naive attention fallback only supports attn_mask.")
        bs, seq_len, _ = query.size()
        k_len = key.size(1)
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)
        if self.qk_norm:
            dtype = q.dtype
            q = self.q_norm(q).to(dtype)
            k = self.k_norm(k).to(dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if rotary_emb is not None:
            q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None and attn_mask.shape[-2:] == scores.shape[-2:]:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)

    original_mha_forward = MultiheadAttention.forward
    MultiheadAttention.forward = _naive_attention_forward
    try:
        model.fwd(data=data, times=times, input_len=1)
    finally:
        MultiheadAttention.forward = original_mha_forward

    if recorded.get("spatial_tokens") != expected:
        raise AssertionError(f"spatial_tokens mismatch: got {recorded.get('spatial_tokens')}, expected {expected}")

    print("Spatial tokens plumbing check passed.")


if __name__ == "__main__":
    main()
