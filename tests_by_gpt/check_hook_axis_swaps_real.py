import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.multiscale_bcat import MultiscaleBCAT  # noqa: E402
from models.attention_utils import MultiheadAttention  # noqa: E402


def _get_time_embeddings(embedder, times: torch.Tensor) -> torch.Tensor:
    if getattr(embedder, "time_embed_type", None) == "continuous":
        return embedder.time_proj(times)
    return embedder.time_embeddings[:, : times.size(1)]


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    model = MultiscaleBCAT(model_cfg, x_num=x_num, max_output_dim=max_output_dim).float()
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    spatial_tokens = model_cfg.embedder.patch_num ** 2
    t_total = 3
    batch_size = 1
    data = torch.zeros(batch_size, t_total, x_num, x_num, max_output_dim, dtype=torch.float32)
    times = torch.arange(t_total, dtype=torch.float32).view(1, t_total, 1)

    captured = {}

    def pre_decode_hook(_module, inputs):
        captured["tokens"] = inputs[0].detach().cpu()

    handle = model.embedder.post_proj.register_forward_pre_hook(pre_decode_hook)
    def _run_forward():
        with torch.no_grad():
            with sdpa_kernel(SDPBackend.MATH):
                model.fwd(data=data, times=times, input_len=1)

    try:
        _run_forward()
    except RuntimeError as exc:
        if "scaled_dot_product_attention" not in str(exc):
            raise

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
            if attn_mask is not None:
                expected = (scores.size(-2), scores.size(-1))
                mask_shape = attn_mask.shape[-2:]
                if mask_shape == expected:
                    if attn_mask.dim() == 2:
                        scores = scores + attn_mask
                    elif attn_mask.dim() == 3:
                        scores = scores + attn_mask.unsqueeze(1)
                    elif attn_mask.dim() == 4:
                        scores = scores + attn_mask
                    else:
                        raise RuntimeError(f"Unsupported attn_mask dims: {attn_mask.dim()}")
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)
            output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
            return self.out_proj(output)

        original_forward = MultiheadAttention.forward
        MultiheadAttention.forward = _naive_attention_forward
        try:
            _run_forward()
        finally:
            MultiheadAttention.forward = original_forward
    handle.remove()

    tokens = captured["tokens"]  # (b, t*s, d)
    if tokens is None:
        raise AssertionError("Pre-decode tokens were not captured.")

    time_emb = _get_time_embeddings(model.embedder, times).detach().cpu()
    if time_emb.dim() == 4:
        time_emb = time_emb.squeeze(0).squeeze(1)
    elif time_emb.dim() == 3:
        time_emb = time_emb.squeeze(0)
    token_flat = tokens[0]
    token_norm = token_flat / (token_flat.norm(dim=-1, keepdim=True) + 1e-8)
    time_norm = time_emb / (time_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sims = token_norm @ time_norm.T  # (t*s, t)
    pred_time = sims.argmax(dim=-1)

    total = token_flat.size(0)
    t_major = torch.arange(total) // spatial_tokens
    s_major = torch.arange(total) % t_total
    t_major_acc = (pred_time == t_major).float().mean().item()
    s_major_acc = (pred_time == s_major).float().mean().item()

    print(f"Time-embedding alignment: t-major={t_major_acc:.3f}, s-major={s_major_acc:.3f}")
    print("Hook-based real-forward check complete.")


if __name__ == "__main__":
    main()
