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
    t_total = 3
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

    time_emb = _get_time_embeddings(model.embedder, times).detach().cpu()
    layer_scores = []

    def _layer_hook(_module, _inputs, outputs):
        z_fast, _ = outputs
        layer_scores.append(_alignment(z_fast.detach().cpu(), time_emb, spatial_tokens))

    def _run_with_overrides(label: str, diagonal_mask: bool, disable_self: bool, disable_cross: bool) -> None:
        original_layer_forwards = []
        if disable_self or disable_cross:
            for layer in model.transformer.layers:
                original_layer_forwards.append((layer.fast_mixer.forward, layer.slow_mixer.forward))

                def _make_forward(mixer, drop_self: bool, drop_cross: bool):
                    split_dim = mixer.split_dim
                    f_d1 = mixer.split_sizes[0]
                    f_d2 = mixer.split_sizes[1]

                    def forward(x, y=None, **kwargs):
                        x1 = x.narrow(split_dim, 0, f_d1)
                        x2 = x.narrow(split_dim, f_d1, f_d2)
                        if drop_self:
                            y1 = torch.zeros_like(x1)
                        else:
                            y1 = mixer.self_branch(x1, **kwargs)
                        if drop_cross:
                            y2 = torch.zeros_like(x2)
                        else:
                            y2 = mixer.cross_branch(x2, y=y, **kwargs)
                        return torch.cat([y1, y2], dim=split_dim)

                    return forward

                layer.fast_mixer.forward = _make_forward(layer.fast_mixer, drop_self=disable_self, drop_cross=disable_cross)
                layer.slow_mixer.forward = _make_forward(layer.slow_mixer, drop_self=disable_self, drop_cross=disable_cross)

        original_fwd = model.fwd
        if diagonal_mask:
            seq_len = t_total * spatial_tokens
            attn_mask = torch.full((seq_len, seq_len), float("-inf"))
            attn_mask.fill_diagonal_(0.0)

            def wrapped_fwd(*args, **kwargs):
                kwargs["fast_attn_mask"] = attn_mask
                return original_fwd(*args, **kwargs)

            model.fwd = wrapped_fwd

        layer_scores.clear()
        hooks = [layer.register_forward_hook(_layer_hook) for layer in model.transformer.layers]
        try:
            with torch.no_grad():
                model.fwd(data=data, times=times, input_len=1)
        finally:
            for h in hooks:
                h.remove()
            if diagonal_mask:
                model.fwd = original_fwd
            if disable_self or disable_cross:
                for layer, (fwd_fast, fwd_slow) in zip(model.transformer.layers, original_layer_forwards):
                    layer.fast_mixer.forward = fwd_fast
                    layer.slow_mixer.forward = fwd_slow

        print(label)
        for i, score in enumerate(layer_scores):
            print(f"  layer_{i}: t-major={score:.3f}")

    _run_with_overrides("baseline", diagonal_mask=False, disable_self=False, disable_cross=False)
    _run_with_overrides("diagonal mask", diagonal_mask=True, disable_self=False, disable_cross=False)
    _run_with_overrides("self-only", diagonal_mask=False, disable_self=False, disable_cross=True)
    _run_with_overrides("cross-only", diagonal_mask=False, disable_self=True, disable_cross=False)

    print("Layer alignment trace complete.")


if __name__ == "__main__":
    main()
