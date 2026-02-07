import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.attention_utils import MultiheadAttention  # noqa: E402
from models.multiscale_bcat import MultiscaleBCAT  # noqa: E402


def _get_time_embeddings(embedder, times: torch.Tensor) -> torch.Tensor:
    if getattr(embedder, "time_embed_type", None) == "continuous":
        return embedder.time_proj(times)
    return embedder.time_embeddings[:, : times.size(1)]


def _time_emb_at_indices(embedder, times: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if getattr(embedder, "time_embed_type", None) == "continuous":
        return embedder.time_proj(times[:, indices])
    return embedder.time_embeddings[:, indices]


def _alignment(tokens: torch.Tensor, time_emb: torch.Tensor, spatial_tokens: int) -> float:
    total = tokens.size(1)
    if total % spatial_tokens != 0:
        return float("nan")
    time_emb = time_emb.squeeze(0)
    if time_emb.dim() == 4:
        time_emb = time_emb.squeeze(1)
    if time_emb.dim() == 3:
        time_emb = time_emb.squeeze(1)
    token_norm = tokens[0] / (tokens[0].norm(dim=-1, keepdim=True) + 1e-8)
    time_norm = time_emb / (time_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sims = token_norm @ time_norm.T
    pred_time = sims.argmax(dim=-1)
    t_major = torch.arange(total) // spatial_tokens
    return (pred_time == t_major).float().mean().item()


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    model = MultiscaleBCAT(model_cfg, x_num=x_num, max_output_dim=max_output_dim).float()
    model.eval()

    spatial_tokens = model_cfg.embedder.patch_num ** 2
    rate = int(model_cfg.rate)
    t_total = 5
    data = torch.zeros(1, t_total, x_num, x_num, max_output_dim, dtype=torch.float32)
    times = torch.arange(t_total, dtype=torch.float32).view(1, t_total, 1)

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

    original_forward = MultiheadAttention.forward
    MultiheadAttention.forward = _naive_attention_forward

    layer_scores = []

    def _layer_hook(_module, inputs, outputs):
        x_slow = inputs[1]
        z_slow = outputs[1]
        slow_len = x_slow.size(1) // spatial_tokens
        idx = torch.arange(slow_len) * rate
        idx = torch.clamp(idx, max=t_total - 1)
        idx[0] = 0
        time_emb_slow = _time_emb_at_indices(model.embedder, times, idx).detach().cpu()
        in_score = _alignment(x_slow.detach().cpu(), time_emb_slow, spatial_tokens)
        out_score = _alignment(z_slow.detach().cpu(), time_emb_slow, spatial_tokens)
        layer_scores.append((in_score, out_score))

    hooks = [layer.register_forward_hook(_layer_hook) for layer in model.transformer.layers]
    try:
        with torch.no_grad():
            model.fwd(data=data, times=times, input_len=1)
    finally:
        for h in hooks:
            h.remove()
        MultiheadAttention.forward = original_forward

    for i, (in_score, out_score) in enumerate(layer_scores):
        print(f"layer_{i}: slow_in={in_score:.3f}, slow_out={out_score:.3f}")

    print("Layer slow alignment trace complete.")


if __name__ == "__main__":
    main()
