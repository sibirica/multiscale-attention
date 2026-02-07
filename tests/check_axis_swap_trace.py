import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.attention_utils import MultiheadAttention  # noqa: E402
from models.multiscale_bcat import MultiscaleBCAT  # noqa: E402
from models.multiscale_utils import lift  # noqa: E402


class IdentityMixer(torch.nn.Module):
    def forward(self, x, y=None, **kwargs):
        return x


def _get_time_embeddings(embedder, times: torch.Tensor) -> torch.Tensor:
    if getattr(embedder, "time_embed_type", None) == "continuous":
        return embedder.time_proj(times)
    return embedder.time_embeddings[:, : times.size(1)]


def _alignment(tokens: torch.Tensor, time_emb: torch.Tensor, spatial_tokens: int) -> tuple[float, float]:
    b, total, _ = tokens.shape
    if total % spatial_tokens != 0:
        return float("nan"), float("nan")
    t_len = total // spatial_tokens
    time_emb = time_emb.squeeze(0)
    if time_emb.dim() == 4:
        time_emb = time_emb.squeeze(1)
    if time_emb.dim() == 3:
        time_emb = time_emb.squeeze(1)

    token_norm = tokens[0] / (tokens[0].norm(dim=-1, keepdim=True) + 1e-8)
    time_norm = time_emb / (time_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sims = token_norm @ time_norm.T  # (t*s, t)
    pred_time = sims.argmax(dim=-1)

    t_major = torch.arange(total) // spatial_tokens
    s_major = torch.arange(total) % t_len
    t_major_acc = (pred_time == t_major).float().mean().item()
    s_major_acc = (pred_time == s_major).float().mean().item()
    return t_major_acc, s_major_acc


def _run_trace(model, data, times, spatial_tokens) -> dict[str, tuple[float, float]]:
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

    captured = {}

    def split_hook(_module, _inputs, outputs):
        captured["x_fast"], captured["x_slow"] = outputs

    def recombine_hook(_module, inputs, output):
        if len(inputs) >= 2:
            captured["z_fast"], captured["z_slow"] = inputs[0], inputs[1]
            z_slow_lift = lift(
                inputs[1],
                rate=_module.rate,
                time_dim=_module.time_dim,
                spatial_tokens=_module.spatial_tokens,
            )
            if z_slow_lift.size(_module.time_dim) > inputs[0].size(_module.time_dim):
                z_slow_lift = z_slow_lift.narrow(_module.time_dim, 0, inputs[0].size(_module.time_dim))
            captured["z_slow_lift"] = z_slow_lift
        captured["recombined"] = output

    def pre_decode_hook(_module, inputs):
        captured["pre_decode"] = inputs[0].detach().cpu()

    h1 = model.transformer.split_encoder.register_forward_hook(split_hook)
    h2 = model.transformer.recombine_decoder.register_forward_hook(recombine_hook)
    h3 = model.embedder.post_proj.register_forward_pre_hook(pre_decode_hook)

    try:
        with torch.no_grad():
            model.fwd(data=data, times=times, input_len=1)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        MultiheadAttention.forward = original_forward

    time_emb = _get_time_embeddings(model.embedder, times).detach().cpu()
    results = {}
    for name in ("x_fast", "z_fast", "z_slow", "z_slow_lift", "recombined", "pre_decode"):
        tokens = captured.get(name)
        if tokens is None:
            continue
        results[name] = _alignment(tokens.detach().cpu(), time_emb, spatial_tokens)
    return results


def _print_results(label: str, results: dict[str, tuple[float, float]]) -> None:
    print(label)
    for name in ("x_fast", "z_fast", "z_slow", "z_slow_lift", "recombined", "pre_decode"):
        if name not in results:
            continue
        t_acc, s_acc = results[name]
        print(f"  {name}: t-major={t_acc:.3f}, s-major={s_acc:.3f}")


def main() -> None:
    model_cfg = OmegaConf.load(ROOT / "src" / "configs" / "model" / "multiscale_bcat.yaml")
    data_cfg = OmegaConf.load(ROOT / "src" / "configs" / "data" / "fluids_arena.yaml")

    x_num = int(data_cfg.x_num)
    max_output_dim = int(data_cfg.max_output_dimension)
    spatial_tokens = model_cfg.embedder.patch_num ** 2

    t_total = 3
    batch_size = 1
    data = torch.zeros(batch_size, t_total, x_num, x_num, max_output_dim, dtype=torch.float32)
    times = torch.arange(t_total, dtype=torch.float32).view(1, t_total, 1)

    model = MultiscaleBCAT(model_cfg, x_num=x_num, max_output_dim=max_output_dim).float()
    model.eval()
    results_full = _run_trace(model, data, times, spatial_tokens)
    _print_results("Trace with mixers", results_full)

    for layer in model.transformer.layers:
        layer.fast_mixer = IdentityMixer()
        layer.slow_mixer = IdentityMixer()

    results_no_mix = _run_trace(model, data, times, spatial_tokens)
    _print_results("Trace without mixers", results_no_mix)

    print("Axis swap trace complete.")


if __name__ == "__main__":
    main()
