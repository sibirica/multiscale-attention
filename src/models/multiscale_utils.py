from typing import Literal, Optional
import torch
import torch.nn as nn

# wandb for logging
import torch.distributed as dist
import wandb

from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding

from .attention_utils import FFN, get_activation, MultiheadAttention, MultiheadFlexAttention, _get_clones
from .kv_cache import KVCache


class AsymmetricFFN(nn.Module):  ### Note: not currently used
    def __init__(self, in_dim, hidden_dim, out_dim, act="gelu", dropout=0):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)

        if act.endswith("glu"):
            self.fc_gate = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_gate = None

        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if self.fc_gate is None:
            return self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            return self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))


def lift(  ### Note: not currently used
    s: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    ffn: Optional[nn.Module] = None,
    spatial_tokens: int = 1,
) -> torch.Tensor:
    """
    Lift a slow sequence to the fast time scale by repeating each timestep `rate` times
    along the specified time dimension.

    Args:
        s: slow sequence tensor. Typical shape: [B, T_slow, D_slow] if time_dim=1.
        rate: positive integer repetition factor; output time length becomes rate * T_slow.
        time_dim: index of the time axis in `s`.
        ffn: optional pointwise feature mapper applied BEFORE repetition.
             If provided, it should map [..., D_slow] -> [..., D_lift].
    Returns:
        out: lifted sequence.
             - If ffn is None: shape is s with time length multiplied by `rate`,
               e.g., [B, rate*T_slow, D_slow].
             - If ffn is provided: feature dim follows the module output,
               e.g., [B, rate*T_slow, D_lift].
    """
    out = s
    if ffn is not None:
        out = ffn(out)
    if spatial_tokens == 1:
        return torch.repeat_interleave(out, repeats=rate, dim=time_dim)

    x = out.movedim(time_dim, 1)
    t = x.shape[1]
    if t % spatial_tokens != 0:
        raise ValueError("lift expects 'time' length divisible by spatial_tokens.")

    t_len = t // spatial_tokens
    # [B, T*S, D] -> [B, T, S, D]
    x = rearrange(x, "b (t s) d -> b t s d", t=t_len, s=spatial_tokens)
    x = torch.repeat_interleave(x, repeats=rate, dim=1)
    # [B, T*R, S, D] -> [B, (T*R*S), D]
    x = rearrange(x, "b t s d -> b (t s) d")
    return x.movedim(1, time_dim)


class PoolFFN(nn.Module):
    """
    Feed-forward mapper from a fast block of shape [..., rate, C_in] to a single slow step [..., C_out].
    Internally flattens the last two dims to [..., rate*C_in], applies MLP, and returns [..., C_out].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        rate: int,
        act: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rate = rate

        input_dim = rate * in_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if act.endswith("glu"):
            self.fc_gate = nn.Linear(input_dim, hidden_dim)
        else:
            self.fc_gate = None
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        # x_blocks: [..., rate, C_in]
        last_two = x_blocks.shape[-2:]
        if last_two != (self.rate, self.in_dim):
            raise ValueError(f"Expected trailing shape ({self.rate}, {self.in_dim}), got {last_two}.")
        x_flat = rearrange(x_blocks, "... r c -> ... (r c)", r=self.rate, c=self.in_dim)
        if self.fc_gate is None:
            y = self.fc2(self.dropout(self.activation(self.fc1(x_flat))))
        else:
            y = self.fc2(self.dropout(self.activation(self.fc1(x_flat), self.fc_gate(x_flat))))
        return y


def pool(
    f: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    pool_ffn: Optional[PoolFFN] = None,
    spatial_tokens: int = 1,
) -> torch.Tensor:
    """
    Block pooling from fast to slow time scale, no shift:
      s_0 = FFN(f_0, ..., f_{rate-1})
      s_1 = FFN(f_{rate}, ..., f_{2*rate-1})
      ...
      s_{K-1} = FFN(f_{(K-1)*rate}, ..., f_{K*rate-1})
    where K = floor(T_fast / rate). A trailing partial block (if any) is
    dropped because the only slow position that would summarize it is
    never read by any causal fast token (fast_to_slow's mask requires
    f_t >= (s+1)*rate - 1, which is unsatisfiable for that trailing s).

    Causality of slow_s is enforced via the cross-attention masks rather
    than via shifting the pool output: slow_to_fast restricts s_s to read
    f_<=(s+1)*rate-1, and fast_to_slow restricts the consumers of s_s to
    fast indices t >= (s+1)*rate-1.

    Args:
        f: fast sequence tensor. Typical shape: [B, T_fast, D_fast] if time_dim=1.
        rate: positive integer; number of fast steps per slow step.
        time_dim: index of the time axis in `f`.
        pool_ffn: required module that maps blocks of shape [..., rate, D_fast] -> [..., D_pool].
                  For example, PoolFFN(in_dim=D_fast, out_dim=D_pool, hidden_dim=..., rate=rate).
    Returns:
        out: pooled slow sequence with K = floor(T_fast / rate) blocks.
             If T_fast < rate (degenerate), returns a single zero slow
             slot so that downstream attention shapes remain valid.
             shape is [B, K, D_pool] if time_dim=1; time length is T_fast // rate.
    """
    # Move time axis to dim=1 for convenience
    x = f.movedim(time_dim, 1)
    b, t, c = x.shape[0], x.shape[1], x.shape[-1]

    if pool_ffn is None:
        raise ValueError("pool_ffn must be provided.")
    out_dim = getattr(pool_ffn, "out_dim", c)
    if spatial_tokens > 1:
        if t % spatial_tokens != 0:
            raise ValueError("pool expects time length divisible by spatial_tokens.")
        t_len = t // spatial_tokens
        num_blocks = t_len // rate
    else:
        num_blocks = t // rate

    if num_blocks == 0:
        zero = torch.zeros(b, spatial_tokens, out_dim, dtype=x.dtype, device=x.device)
        return zero.movedim(1, time_dim)

    if spatial_tokens > 1:
        t_eff = num_blocks * rate
        x_trim = x[:, : t_eff * spatial_tokens, :]
        x_blocks = rearrange(
            x_trim,
            "b (bk r s) d -> b bk s r d",
            bk=num_blocks,
            r=rate,
            s=spatial_tokens,
        )
    else:
        t_eff = num_blocks * rate
        x_trim = x[:, :t_eff, :]
        # [B, T, D] -> [B, Bk, R, D]
        x_blocks = rearrange(
            x_trim,
            "b (bk r) d -> b bk r d",
            bk=num_blocks,
            r=rate,
        )

    s_blocks = pool_ffn(x_blocks)  # [B, Bk, D_pool] or [B, Bk, S, D_pool]
    if s_blocks.dim() == 3:
        s_blocks = s_blocks.unsqueeze(2)  # [B, Bk, 1, D_pool] -> uniform shape
    s = rearrange(s_blocks, "b t s d -> b (t s) d")
    return s.movedim(1, time_dim)


class SplitEncoder(nn.Module):
    """
    Wrapper that computes a fast embedding via the embedder,
    then pools it to a slow embedding. Returns (fast, slow).
    """

    def __init__(
        self,
        embedder: nn.Module,
        rate: int,
        pool_ffn: nn.Module,
        time_dim: int = 1,
        spatial_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.rate = rate
        self.pool_ffn = pool_ffn
        self.time_dim = time_dim
        self.spatial_tokens = spatial_tokens

    def forward(self, *args, skip_len: int = 0, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        enc_kw = dict(kwargs)
        enc_kw["skip_len"] = skip_len
        f = self.embedder.encode(*args, **enc_kw)
        # No pad_for_slow: trailing partial blocks are dropped inside pool()
        # because the corresponding slow slot is never read by a causal fast token.
        s = pool(
            f,
            rate=self.rate,
            time_dim=self.time_dim,
            pool_ffn=self.pool_ffn,
            spatial_tokens=self.spatial_tokens,
        )
        return f, s


# Note: is_causal is somewhat redundant since the masks are causal anyway,
# but it seems like we're setting it to False everywhere anyway
class TwoScaleTransformerEncoderLayer(nn.Module):
    """
    Two-scale encoder layer with separate self- and cross-attention.
    """

    def __init__(
        self,
        fast_embed_dim: int,
        slow_embed_dim: int,
        num_heads: int,
        rate: int,
        dim_ffn: int,
        dropout: float = 0.0,
        act: str = "gelu",
        bias: bool = True,
        qk_norm: bool = False,
        flex_attn: bool = False,
        shared_scale_ffn: bool = False,
        disable_slow_scale: bool = False,  ### FLAG FOR DEBUGGING W/O SLOW SCALE
        norm: type[nn.Module] = nn.LayerNorm,
        peri_ln: bool = False,
    ) -> None:
        super().__init__()
        self.fast_embed_dim = fast_embed_dim
        self.slow_embed_dim = slow_embed_dim
        self.rate = rate

        self.disable_slow_scale = disable_slow_scale  # True

        if fast_embed_dim != slow_embed_dim:
            raise ValueError("Cross-attention requires fast_embed_dim == slow_embed_dim.")

        if flex_attn:
            SelfMHA = MultiheadFlexAttention
            CrossMHA = MultiheadFlexAttention
        else:
            SelfMHA = MultiheadAttention
            CrossMHA = MultiheadAttention
        self.self_attn_fast = SelfMHA(
            fast_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
        )
        self.self_attn_slow = SelfMHA(
            slow_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
        )
        if self.disable_slow_scale:  # no bias from slow scale either
            self.cross_attn_fast = CrossMHA(
                fast_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=False,
                qk_norm=qk_norm,
            )
        else:
            self.cross_attn_fast = CrossMHA(
                fast_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                qk_norm=qk_norm,
            )
        self.cross_attn_slow = CrossMHA(
            slow_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
        )

        self.norm_fast_attn = norm(fast_embed_dim)
        self.norm_slow_attn = norm(slow_embed_dim)
        self.norm_fast_ffn = norm(fast_embed_dim)
        self.norm_slow_ffn = norm(slow_embed_dim)
        self.peri_ln = peri_ln
        self.norm_fast_self_attn_out = norm(fast_embed_dim) if peri_ln else nn.Identity()
        self.norm_fast_cross_attn_out = norm(fast_embed_dim) if peri_ln else nn.Identity()
        self.norm_slow_self_attn_out = norm(slow_embed_dim) if peri_ln else nn.Identity()
        self.norm_slow_cross_attn_out = norm(slow_embed_dim) if peri_ln else nn.Identity()
        self.norm_fast_ffn_post = norm(fast_embed_dim) if peri_ln else nn.Identity()
        self.norm_slow_ffn_post = norm(slow_embed_dim) if peri_ln else nn.Identity()

        if shared_scale_ffn:
            shared_ffn = FFN(fast_embed_dim, dim_ffn, act=act, dropout=dropout)
            self.ffn_fast = shared_ffn
            self.ffn_slow = shared_ffn
        else:
            self.ffn_fast = FFN(fast_embed_dim, dim_ffn, act=act, dropout=dropout)
            self.ffn_slow = FFN(slow_embed_dim, dim_ffn, act=act, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x_fast: torch.Tensor,  # [B, L_fast, D_fast]
        x_slow: torch.Tensor,  # [B, L_slow, D_slow]
        masks: dict,
        is_causal: bool = False,
        spatial_tokens: int = 1,
        rotary_emb=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast/slow attention block with residuals.
        Shapes:
          x_fast: [B, L_fast, D_fast], where L_fast = T_fast * spatial_tokens
          x_slow: [B, L_slow, D_slow], where L_slow = T_slow * spatial_tokens
          T_slow = ceil(T_fast / rate)
        Mask dict:
          attn_* entries are dense masks (float or bool) for SDPA
          block_* entries are BlockMask objects for flex attention
        """
        Bf, Lf, Df = x_fast.shape[0], x_fast.shape[1], x_fast.shape[-1]
        Bs, Ls, Ds = x_slow.shape[0], x_slow.shape[1], x_slow.shape[-1]

        # Validate batch and feature dims
        if Bf != Bs:
            raise ValueError(f"Batch sizes must match: fast {Bf} vs slow {Bs}.")
        if Df != self.fast_embed_dim:
            raise ValueError(f"Fast feature dim must equal fast_embed_dim={self.fast_embed_dim}: got {Df}.")
        if Ds != self.slow_embed_dim:
            raise ValueError(f"Slow feature dim must equal slow_embed_dim={self.slow_embed_dim}: got {Ds}.")
        if Lf % spatial_tokens != 0 or Ls % spatial_tokens != 0:
            raise ValueError("Sequence lengths must be divisible by spatial_tokens.")
        fast_time = Lf // spatial_tokens
        slow_time = Ls // spatial_tokens
        expected_slow_time = max(1, fast_time // self.rate)
        if slow_time != expected_slow_time:
            raise ValueError(f"Temporal sizes mismatch: L_fast={Lf}, rate={self.rate}, but L_slow={Ls}.")

        if self.disable_slow_scale:
            x_slow *= 0.0

        # Residual for attention block
        x_f_res = x_fast
        x_s_res = x_slow

        # Pre-norm inputs for attention
        x_f = self.norm_fast_attn(x_fast)
        x_s = self.norm_slow_attn(x_slow)

        # Expected keys:
        # Either:
        #   fast_self_attn_mask: [L_fast, L_fast]
        #   slow_self_attn_mask: [L_slow, L_slow]
        #   fast_to_slow_attn_mask: [L_fast, L_slow]
        #   slow_to_fast_attn_mask: [L_slow, L_fast]
        # Or:
        #   fast_block_mask: BlockMask(L_fast, L_fast)
        #   slow_block_mask: BlockMask(L_slow, L_slow)
        #   fast_to_slow_block_mask: BlockMask(L_fast, L_slow)
        #   slow_to_fast_block_mask: BlockMask(L_slow, L_fast)
        # masks is required and should contain the keys documented above
        # Self attention (causal on each stream) + cross attention (paired stream)
        # z_f1, z_f2: [B, L_fast, D_fast], z_s1, z_s2: [B, L_slow, D_slow]
        z_f1, z_f2 = self._apply_self_cross(
            x=x_f,
            y=x_s,
            self_attn=self.self_attn_fast,
            cross_attn=self.cross_attn_fast,
            masks=masks,
            self_attn_mask_key="fast_self_attn_mask",
            self_block_mask_key="fast_block_mask",
            cross_attn_mask_key="fast_to_slow_attn_mask",
            cross_block_mask_key="fast_to_slow_block_mask",
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        z_s1, z_s2 = self._apply_self_cross(
            x=x_s,
            y=x_f,
            self_attn=self.self_attn_slow,
            cross_attn=self.cross_attn_slow,
            masks=masks,
            self_attn_mask_key="slow_self_attn_mask",
            self_block_mask_key="slow_block_mask",
            cross_attn_mask_key="slow_to_fast_attn_mask",
            cross_block_mask_key="slow_to_fast_block_mask",
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )

        # Residual add for attention outputs
        if self.peri_ln:
            z_f1 = self.norm_fast_self_attn_out(z_f1)
            z_f2 = self.norm_fast_cross_attn_out(z_f2)
            z_s1 = self.norm_slow_self_attn_out(z_s1)
            z_s2 = self.norm_slow_cross_attn_out(z_s2)
        attn_fast_out = z_f1 + z_f2
        attn_slow_out = z_s1 + z_s2
        fast_attn_residual = x_f_res + attn_fast_out
        slow_attn_residual = x_s_res + attn_slow_out

        # FFN with residual (per stream)
        fast_ffn_input = self.norm_fast_ffn(fast_attn_residual)
        slow_ffn_input = self.norm_slow_ffn(slow_attn_residual)

        fast_ffn_out = self.dropout(self.ffn_fast(fast_ffn_input))
        slow_ffn_out = self.dropout(self.ffn_slow(slow_ffn_input))

        if self.peri_ln:
            fast_ffn_out = self.norm_fast_ffn_post(fast_ffn_out)
            slow_ffn_out = self.norm_slow_ffn_post(slow_ffn_out)

        x_fast = fast_attn_residual + fast_ffn_out
        x_slow = slow_attn_residual + slow_ffn_out
        return x_fast, x_slow

    def _apply_self_cross(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        masks: dict,
        self_attn_mask_key: str,
        self_block_mask_key: str,
        cross_attn_mask_key: str,
        cross_block_mask_key: str,
        is_causal: bool,
        rotary_emb=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_self = self_attn(
            x,
            x,
            x,
            attn_mask=masks.get(self_attn_mask_key),
            block_mask=masks.get(self_block_mask_key),
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        z_cross = cross_attn(
            x,
            y,
            y,
            attn_mask=masks.get(cross_attn_mask_key),
            block_mask=masks.get(cross_block_mask_key),
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        return z_self, z_cross

    def _update_kv_cache_only(
        self,
        attn: nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: KVCache,
    ) -> None:
        bs, k_len, _ = key.size()
        k = attn.linear_k(key)
        v = attn.linear_v(value)
        k = k.view(bs, k_len, attn.num_heads, attn.head_dim)
        v = v.view(bs, k_len, attn.num_heads, attn.head_dim)
        if attn.qk_norm:
            k = attn.k_norm(k).to(k.dtype)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        kv_cache.update(k, v)

    def forward_kv_cache_block(
        self,
        kv_slot_base: int,
        fast_incremental_residual: torch.Tensor,
        slow_incremental_residual: torch.Tensor,
        total_fast_tokens: int,
        total_slow_tokens: int,
        incremental_fast_len: int,
        masks: dict,
        kv_cache: KVCache,
        rotary_emb=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One depth of the two-stream stack during KV-cached autoregressive decoding.

        **Lengths (fast axis is flattened patches-per-frame):**

        - ``total_fast_tokens``: full fast prefix length after this step's new tokens join KV
          (`lf` elsewhere in the codebase).
        - ``total_slow_tokens``: full completed slow prefix length after this step's new
          slow tokens join KV.
        - ``incremental_fast_len``: sequence length of ``fast_incremental_residual``
          (= ``xf_step.size(1)`` after BCAT-like ``skip_len`` encoding).

        ``kv_slot_base`` is ``4 * layer_index``:
        slot ``+0`` holds fast self-attention KV, ``+1`` holds fast→slow
        cross-attention KV on the slow stream, ``+2`` holds slow self-attention
        KV, and ``+3`` holds slow→fast cross-attention KV on the fast stream.

        The KV rollout uses dense masks so compiled generation can reuse one
        prebuilt mask dictionary across all decoding steps.

        Rotary is not supported in this path (``rotary_emb`` must be ``None``).


        **Tensor shapes (B=batch):**

        - ``fast_incremental_residual``: ``[B, incremental_fast_len, D_fast]`` (this AR step's fast patches only).
        - ``slow_incremental_residual``: ``[B, incremental_slow_len, D_slow]`` for
          newly completed slow blocks only.
        - Returns ``(fast_out, slow_out)`` with the same ``[B, *, D]`` leading dimensions as the inputs above.
        """
        if rotary_emb is not None:
            raise RuntimeError("Multiscale KV inference does not support rotary embeddings.")

        if self.disable_slow_scale:
            slow_incremental_residual = slow_incremental_residual * 0.0

        incremental_slow_len = slow_incremental_residual.size(1)

        # Same pre-norm layout as ``forward()``; activations stay ``[B, L, D]`` along each stream.
        fast_for_attn = self.norm_fast_attn(fast_incremental_residual)  # [B, incremental_fast_len, D_fast]
        slow_for_attn = self.norm_slow_attn(slow_incremental_residual)

        # Dense uses float additive masks for SDPA; flex uses BlockMask objects.
        dense_fast_self = masks.get("fast_self_attn_mask")
        dense_fast_to_slow = masks.get("fast_to_slow_attn_mask")
        dense_slow_self = masks.get("slow_self_attn_mask")
        dense_slow_to_fast = masks.get("slow_to_fast_attn_mask")
        block_fast_self = masks.get("fast_block_mask")
        block_fast_to_slow = masks.get("fast_to_slow_block_mask")
        block_slow_self = masks.get("slow_block_mask")
        block_slow_to_fast = masks.get("slow_to_fast_block_mask")
        fast_kv_len = (
            dense_fast_self.size(1) if dense_fast_self is not None and kv_cache.return_full_cache else total_fast_tokens
        )
        slow_kv_len = (
            dense_fast_to_slow.size(1)
            if dense_fast_to_slow is not None and kv_cache.return_full_cache
            else total_slow_tokens
        )

        # --- Fast self-attention (writes KV slot ``kv_slot_base``). ---
        attn_mask_fast_self = None
        if dense_fast_self is not None:
            attn_mask_fast_self = dense_fast_self[
                total_fast_tokens - incremental_fast_len : total_fast_tokens,
                :fast_kv_len,
            ]

        kv_cache.set_layer(kv_slot_base)
        attn_delta_fast_self = self.self_attn_fast(
            fast_for_attn,
            fast_for_attn,
            fast_for_attn,
            attn_mask=attn_mask_fast_self,
            block_mask=block_fast_self,
            is_causal=False,
            rotary_emb=None,
            cache=kv_cache,
        )

        # --- Fast attends to pooled slow (writes/reads KV slot ``kv_slot_base + 1``). ---
        if total_slow_tokens == 0:
            attn_delta_fast_from_slow = torch.zeros_like(fast_incremental_residual)
        else:
            attn_mask_fast_to_slow = None
            if dense_fast_to_slow is not None:
                attn_mask_fast_to_slow = dense_fast_to_slow[
                    total_fast_tokens - incremental_fast_len : total_fast_tokens,
                    :slow_kv_len,
                ]
            kv_cache.set_layer(kv_slot_base + 1)
            attn_delta_fast_from_slow = self.cross_attn_fast(
                fast_for_attn,
                slow_for_attn,
                slow_for_attn,
                attn_mask=attn_mask_fast_to_slow,
                block_mask=block_fast_to_slow,
                is_causal=False,
                rotary_emb=None,
                cache=kv_cache,
            )

        if incremental_slow_len == 0:
            kv_cache.set_layer(kv_slot_base + 3)
            self._update_kv_cache_only(self.cross_attn_slow, fast_for_attn, fast_for_attn, kv_cache)
            slow_attn_residual = slow_incremental_residual
        else:
            # --- Slow self-attention (writes/reads KV slot ``kv_slot_base + 2``). ---
            attn_mask_slow_self = None
            if dense_slow_self is not None:
                attn_mask_slow_self = dense_slow_self[
                    total_slow_tokens - incremental_slow_len : total_slow_tokens,
                    :slow_kv_len,
                ]
            kv_cache.set_layer(kv_slot_base + 2)
            attn_delta_slow_self = self.self_attn_slow(
                slow_for_attn,
                slow_for_attn,
                slow_for_attn,
                attn_mask=attn_mask_slow_self,
                block_mask=block_slow_self,
                is_causal=False,
                rotary_emb=None,
                cache=kv_cache,
            )

            # --- Slow queries read fast K/V (writes/reads KV slot ``kv_slot_base + 3``). ---
            kv_cache.set_layer(kv_slot_base + 3)
            attn_mask_slow_to_fast = None
            if dense_slow_to_fast is not None:
                attn_mask_slow_to_fast = dense_slow_to_fast[
                    total_slow_tokens - incremental_slow_len : total_slow_tokens,
                    :fast_kv_len,
                ]
            attn_delta_slow_from_fast = self.cross_attn_slow(
                slow_for_attn,
                fast_for_attn,
                fast_for_attn,
                attn_mask=attn_mask_slow_to_fast,
                block_mask=block_slow_to_fast,
                is_causal=False,
                rotary_emb=None,
                cache=kv_cache,
            )
            if self.peri_ln:
                attn_delta_slow_self = self.norm_slow_self_attn_out(attn_delta_slow_self)
                attn_delta_slow_from_fast = self.norm_slow_cross_attn_out(attn_delta_slow_from_fast)
            slow_attn_out = attn_delta_slow_self + attn_delta_slow_from_fast
            slow_attn_residual = slow_incremental_residual + slow_attn_out

        # Residual + FFN ordering matches ``forward()`` branch on this layer.
        if self.peri_ln:
            attn_delta_fast_self = self.norm_fast_self_attn_out(attn_delta_fast_self)
            attn_delta_fast_from_slow = self.norm_fast_cross_attn_out(attn_delta_fast_from_slow)
        fast_attn_out = attn_delta_fast_self + attn_delta_fast_from_slow
        fast_attn_residual = fast_incremental_residual + fast_attn_out

        fast_ffn_input = self.norm_fast_ffn(fast_attn_residual)
        fast_ffn_out = self.dropout(self.ffn_fast(fast_ffn_input))
        if self.peri_ln:
            fast_ffn_out = self.norm_fast_ffn_post(fast_ffn_out)
        fast_after_ffn = fast_attn_residual + fast_ffn_out

        slow_ffn_input = self.norm_slow_ffn(slow_attn_residual)
        slow_ffn_out = self.dropout(self.ffn_slow(slow_ffn_input))
        if self.peri_ln:
            slow_ffn_out = self.norm_slow_ffn_post(slow_ffn_out)
        slow_after_ffn = slow_attn_residual + slow_ffn_out

        return fast_after_ffn, slow_after_ffn


class Decoder(nn.Module):
    """Common interface for recombination decoders."""


class RecombineDecoder(Decoder):  ### note: not currently used
    """
    Recombine (z_fast, z_slow) into a fast-length sequence via channel concatenation
    after lifting z_slow to the fast time scale, followed by a 2-layer MLP to project
    back to fast_embed_dim.
    """

    def __init__(
        self,
        fast_embed_dim: int,
        slow_embed_dim: int,
        rate: int,
        hidden_dim: int,
        embedder: nn.Module,
        lift_ffn: Optional[nn.Module] = None,
        lift_dim: Optional[int] = None,
        lift_norm: Optional[type[nn.Module]] = None,
        act: str = "gelu",
        norm: type[nn.Module] = nn.LayerNorm,
        dropout: float = 0.0,
        time_dim: int = 1,
        spatial_tokens: int = 1,
        log_norms: bool = False,
        log_norms_mode: Literal["inference", "always"] = "always",
        log_once: bool = True,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.time_dim = time_dim
        self.spatial_tokens = spatial_tokens
        self.lift_ffn = lift_ffn
        self.slow_embed_dim = slow_embed_dim
        self.embedder = embedder

        # slow/fast norm logging
        self.log_norms = log_norms
        self.log_norms_mode = log_norms_mode
        self.log_once = log_once
        self._has_logged = False
        if self.log_norms_mode not in {"inference", "always"}:
            raise ValueError("log_norms_mode must be either 'inference' or 'always'.")

        if lift_dim is None:
            if lift_ffn is None:
                lift_dim = slow_embed_dim
            elif hasattr(lift_ffn, "out_features"):
                lift_dim = lift_ffn.out_features
            elif hasattr(lift_ffn, "out_dim"):
                lift_dim = lift_ffn.out_dim
        self.lift_norm = lift_norm(lift_dim) if lift_norm is not None else None
        in_dim = fast_embed_dim + lift_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if act.endswith("glu"):
            self.fc_gate = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_gate = None
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, fast_embed_dim)
        # self.norm_fast = nn.LayerNorm(fast_embed_dim)
        # self.norm_slow = nn.LayerNorm(slow_embed_dim)
        self.norm_fast = norm(in_dim)  # nn.LayerNorm(in_dim)

    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._has_logged = False
        # Align time scales by lifting z_slow to the fast length
        slow_feedback_off = False  # debug mode: pull zeros from slow scale
        # if slow_feedback_off:
        #     z_slow_lift = z_fast.new_zeros(*z_fast.shape[:-1], self.slow_embed_dim)
        # else:
        z_slow_lift = lift(
            z_slow,
            rate=self.rate,
            time_dim=self.time_dim,
            ffn=self.lift_ffn,
            spatial_tokens=self.spatial_tokens,
        )  # [..., L_fast, slow_embed_dim]

        if self.lift_norm is not None:
            z_slow_lift = self.lift_norm(z_slow_lift)

        if slow_feedback_off:
            z_slow_lift *= 0.0

        if z_slow_lift.size(self.time_dim) > z_fast.size(self.time_dim):
            z_slow_lift = z_slow_lift.narrow(self.time_dim, 0, z_fast.size(self.time_dim))
        if z_slow_lift.size(self.time_dim) != z_fast.size(self.time_dim):
            raise ValueError("RecombineDecoder: time dimensions do not match after lift().")
        if self._should_log_norms():
            wandb.log(
                {
                    "debug/recombine_z_fast_l2_norm": z_fast.detach().norm(dim=-1).mean().item(),
                    "debug/recombine_z_slow_lift_l2_norm": z_slow_lift.detach().norm(dim=-1).mean().item(),
                },
                commit=False,
            )
            if self.log_once:
                self._has_logged = True
        # z_fast_norm = self.norm_fast(z_fast)
        # z_slow_norm = self.norm_slow(z_slow_lift)
        # x = torch.cat([z_fast_norm, z_slow_norm], dim=-1)
        x = torch.cat([z_fast, z_slow_lift], dim=-1)
        ### NOTE: this uses post-LN norm (https://arxiv.org/pdf/2510.09904)
        x = self.norm_fast(x)
        if self.fc_gate is None:
            y = z_fast + self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            y = z_fast + self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))
        return self.embedder.decode(y)

    def _should_log_norms(self) -> bool:
        if not self.log_norms:
            return False
        if self.log_norms_mode == "inference" and self.training:
            return False
        if self.log_once and self._has_logged:
            return False
        if wandb is None or wandb.run is None:
            return False
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        return True


class FastOnlyRecombineDecoder(Decoder):
    """
    Decoder variant that ignores the slow scale entirely.
    Applies normalization on the fast scale and decodes it.
    """

    def __init__(
        self,
        fast_embed_dim: int,
        embedder: nn.Module,
        norm: type[nn.Module] = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.norm_fast = norm(fast_embed_dim)

    def forward(self, z_fast: torch.Tensor) -> torch.Tensor:
        y = self.norm_fast(z_fast)
        return self.embedder.decode(y)


class TwoScaleTransformerEncoder(nn.Module):
    """
    Stack of TwoScaleTransformerEncoderLayer layers, mirroring CustomTransformerEncoder but
    operating on coupled fast/slow streams. Applies layers sequentially and returns (fast, slow).
    """

    def __init__(
        self,
        encoder_layer: TwoScaleTransformerEncoderLayer,
        num_layers: int,
        # norm_fast: Optional[nn.Module] = None,
        # norm_slow: Optional[nn.Module] = None,
        split_encoder: Optional[SplitEncoder] = None,
        decoder: Optional[Decoder] = None,
        config=None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.rate = encoder_layer.rate
        # self.norm_fast = norm_fast
        # self.norm_slow = norm_slow
        self.split_encoder = split_encoder
        self.decoder = decoder
        if config is not None and config.rotary:
            self.rotary_emb = RotaryEmbedding(dim=config.dim_emb // config.n_head // 2)
            self.rotary = True
        else:
            self.rotary_emb = None
            self.rotary = False

    def forward(
        self,
        *args,
        masks: dict,
        is_causal: bool = False,
        spatial_tokens: int = 1,
        full: bool = True,
        **kwargs,
    ):
        """
        full=True runs the recombined full pipeline.
        full=False runs the multi-branch behavior.
        """
        if full:
            return self.forward_full(
                *args,
                masks=masks,
                is_causal=is_causal,
                spatial_tokens=spatial_tokens,
                rotary_emb=self.rotary_emb if self.rotary else None,
                **kwargs,
            )

        # Multi-branch behavior without recombination.
        # Determine input format
        if "x_fast" in kwargs and "x_slow" in kwargs:
            x_fast, x_slow = kwargs["x_fast"], kwargs["x_slow"]
        elif len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x_fast, x_slow = args[0], args[1]
        else:
            raise ValueError("full=False requires x_fast/x_slow positional or keyword args.")

        return self._apply_layers(
            x_fast,
            x_slow,
            masks=masks,
            is_causal=is_causal,
            spatial_tokens=spatial_tokens,
            rotary_emb=self.rotary_emb if self.rotary else None,
        )

    def forward_full(
        self,
        *args,
        masks: dict,
        is_causal: bool = False,
        spatial_tokens: int = 1,
        rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full pipeline:
          inputs -> SplitEncoder -> (x_fast, x_slow) -> stacked layers -> (z_fast, z_slow) -> RecombineDecoder -> y
        Returns:
          y: [B, L, fast_embed_dim]
        """
        if self.split_encoder is None or self.decoder is None:
            raise ValueError("forward_full requires split_encoder and decoder to be set.")

        x_fast, x_slow = self.split_encoder(*args, **kwargs)
        z_fast, z_slow = self._apply_layers(
            x_fast,
            x_slow,
            masks=masks,
            is_causal=is_causal,
            spatial_tokens=spatial_tokens,
            rotary_emb=rotary_emb,
        )
        match self.decoder:
            case FastOnlyRecombineDecoder():
                y = self.decoder(z_fast)
            case _:
                y = self.decoder(z_fast, z_slow)
        return y

    def _apply_layers(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        masks: dict,
        is_causal: bool = False,
        spatial_tokens: int = 1,
        rotary_emb=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_fast = x_fast
        out_slow = x_slow

        for layer in self.layers:
            out_fast, out_slow = layer(
                out_fast,
                out_slow,
                masks=masks,
                is_causal=is_causal,
                spatial_tokens=spatial_tokens,
                rotary_emb=rotary_emb,
            )

        # drop redundant norms
        # if self.norm_fast is not None:
        #     out_fast = self.norm_fast(out_fast)
        # if self.norm_slow is not None:
        #     out_slow = self.norm_slow(out_slow)

        return out_fast, out_slow

    def forward_rollout_kv_cache(
        self,
        fast_incremental_residual: torch.Tensor,
        total_fast_tokens: int,
        slow_incremental_residual: torch.Tensor,
        total_slow_tokens: int,
        *,
        masks: dict,
        spatial_tokens: int,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Cached autoregressive step: analogous to feeding incremental ``src`` through
        ``CacheCustomTransformerEncoder.forward(src, ..., cache=...)``, but over the
        coupled fast/slow stack. Each stacked layer consumes two KV slots (via
        ``forward_kv_cache_block``), similar in spirit to BCAT consuming one slot per depth.

        Arguments match ``forward_kv_cache_block`` tensors/lengths:
        ``fast_incremental_residual``, ``total_fast_tokens``,
        ``slow_incremental_residual``, ``total_slow_tokens``.

        **Shapes:** ``fast_incremental_residual`` is ``[B, incremental_fast_len, D_fast]``,
        ``slow_incremental_residual`` is ``[B, incremental_slow_len, D_slow]``.
        Returns decoded grid per
        ``spatial_tokens`` tail (see ``out_fast`` slice below).
        """
        if self.rotary:
            raise RuntimeError("forward_rollout_kv_cache requires rotary=0.")

        # ``incremental_fast_len`` matches BCAT ``CacheCustomTransformerEncoderLayer`` ``new_len`` for this step.
        incremental_fast_len = fast_incremental_residual.size(1)
        fast_stream = fast_incremental_residual
        slow_stream = slow_incremental_residual
        for layer_idx, layer in enumerate(self.layers):
            kv_slot_base = 4 * layer_idx
            fast_stream, slow_stream = layer.forward_kv_cache_block(
                kv_slot_base,
                fast_stream,
                slow_stream,
                total_fast_tokens,
                total_slow_tokens,
                incremental_fast_len,
                masks,
                kv_cache,
                rotary_emb=None,
            )

        # Tail is the last spatial block of the fast stream (last predicted frame's patches).
        out_fast = fast_stream[:, -spatial_tokens:, :].contiguous()  # [B, spatial_tokens, D_fast]
        match self.decoder:
            case FastOnlyRecombineDecoder():
                return self.decoder(out_fast)
            case _:
                raise NotImplementedError("forward_rollout_kv_cache supports FastOnlyRecombineDecoder.")


if __name__ == "__main__":
    torch.manual_seed(0)
    from models.multiscale_bcat import (
        build_fast_to_slow_mask,
        build_self_attn_mask,
        build_slow_to_fast_mask,
    )

    batch_size = 2
    fast_embed_dim = 8
    slow_embed_dim = 8
    num_heads = 2
    patch_num = 4
    spatial_tokens = patch_num * patch_num
    time_len_fast = 8
    temporal_rate = 2
    time_len_slow = time_len_fast // temporal_rate
    rate = temporal_rate

    if time_len_fast % temporal_rate != 0:
        raise ValueError("time_len_fast must be divisible by temporal_rate.")

    class IdentityEmbedder(nn.Module):
        def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return x

        def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return x

    encoder_pool_ffn = PoolFFN(
        in_dim=fast_embed_dim,
        out_dim=slow_embed_dim,
        hidden_dim=16,
        rate=rate,
        act="gelu",
        dropout=0.0,
    )
    layer = TwoScaleTransformerEncoderLayer(
        fast_embed_dim=fast_embed_dim,
        slow_embed_dim=slow_embed_dim,
        num_heads=num_heads,
        rate=rate,
        dim_ffn=16,
        dropout=0.0,
        act="gelu",
        bias=True,
        qk_norm=False,
        flex_attn=False,
    )

    split_encoder = SplitEncoder(
        embedder=IdentityEmbedder(),
        rate=rate,
        pool_ffn=encoder_pool_ffn,
        time_dim=1,
        spatial_tokens=spatial_tokens,
    )
    decoder = RecombineDecoder(
        fast_embed_dim=fast_embed_dim,
        slow_embed_dim=slow_embed_dim,
        rate=rate,
        hidden_dim=16,
        act="gelu",
        dropout=0.0,
        time_dim=1,
        spatial_tokens=spatial_tokens,
        embedder=IdentityEmbedder(),
    )
    encoder = TwoScaleTransformerEncoder(
        encoder_layer=layer,
        num_layers=2,
        # norm_fast=nn.LayerNorm(fast_embed_dim),
        # norm_slow=nn.LayerNorm(slow_embed_dim),
        split_encoder=split_encoder,
        decoder=decoder,
    )

    fast_seq_len = time_len_fast * spatial_tokens
    slow_seq_len = time_len_slow * spatial_tokens

    x_fast = torch.randn(batch_size, fast_seq_len, fast_embed_dim)
    x_slow = torch.randn(batch_size, slow_seq_len, slow_embed_dim)

    fast_self_mask = build_self_attn_mask(time_len_fast, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype)
    slow_self_mask = build_self_attn_mask(time_len_slow, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype)
    fast_to_slow_mask = build_fast_to_slow_mask(
        time_len_fast, time_len_slow, rate, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype
    )
    slow_to_fast_mask = build_slow_to_fast_mask(
        time_len_fast, time_len_slow, rate, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype
    )

    z_fast, z_slow = layer(
        x_fast=x_fast,
        x_slow=x_slow,
        masks={
            "fast_self_attn_mask": fast_self_mask,
            "slow_self_attn_mask": slow_self_mask,
            "fast_to_slow_attn_mask": fast_to_slow_mask,
            "slow_to_fast_attn_mask": slow_to_fast_mask,
        },
        is_causal=False,
        spatial_tokens=spatial_tokens,
    )

    out_full = encoder(
        x_fast,
        masks={
            "fast_self_attn_mask": fast_self_mask,
            "slow_self_attn_mask": slow_self_mask,
            "fast_to_slow_attn_mask": fast_to_slow_mask,
            "slow_to_fast_attn_mask": slow_to_fast_mask,
        },
        is_causal=False,
        spatial_tokens=spatial_tokens,
        full=True,
    )

    out_fast, out_slow = encoder(
        x_fast,
        x_slow,
        masks={
            "fast_self_attn_mask": fast_self_mask,
            "slow_self_attn_mask": slow_self_mask,
            "fast_to_slow_attn_mask": fast_to_slow_mask,
            "slow_to_fast_attn_mask": slow_to_fast_mask,
        },
        is_causal=False,
        spatial_tokens=spatial_tokens,
        full=False,
    )

    expected_fast = (batch_size, fast_seq_len, fast_embed_dim)
    expected_slow = (batch_size, slow_seq_len, slow_embed_dim)
    assert z_fast.shape == expected_fast, f"z_fast.shape={z_fast.shape}, expected={expected_fast}"
    assert z_slow.shape == expected_slow, f"z_slow.shape={z_slow.shape}, expected={expected_slow}"
    assert out_full.shape == expected_fast, f"out_full.shape={out_full.shape}, expected={expected_fast}"
    assert out_fast.shape == expected_fast, f"out_fast.shape={out_fast.shape}, expected={expected_fast}"
    assert out_slow.shape == expected_slow, f"out_slow.shape={out_slow.shape}, expected={expected_slow}"

    print("TwoScaleTransformerEncoderLayer z_fast:", z_fast.shape)
    print("TwoScaleTransformerEncoderLayer z_slow:", z_slow.shape)
    print("TwoScaleTransformerEncoder full output:", out_full.shape)
    print("TwoScaleTransformerEncoder out_fast:", out_fast.shape)
    print("TwoScaleTransformerEncoder out_slow:", out_slow.shape)
