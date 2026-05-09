"""
Autoregressive multiscale BCAT model.
"""

from logging import getLogger
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from .attention_utils import DynamicTanh
from .embedder import get_embedder
from .multiscale_utils import (
    #AsymmetricFFN,
    #RecombineDecoder,
    FastOnlyRecombineDecoder,
    PoolFFN,
    SplitEncoder,
    TwoScaleTransformerEncoder,
    TwoScaleTransformerEncoderLayer,
    pool,
)
from .kv_cache import KVCache

# flex ``mask_mod``: when sliding window is disabled, ``_effective_window`` maps
# ``None`` / non-positive ``window`` to a large bound so ``(diff) < window`` does not bind.

_WINDOW_UNBOUNDED = 2**30


def _effective_window(w: Optional[int]) -> int:
    if w is None or w <= 0:
        return _WINDOW_UNBOUNDED
    return int(w)


def _block_self(
    b, h, q_idx, kv_idx, block_size: int, window: Optional[int]
) -> torch.Tensor:
    w = _effective_window(window)
    q_t = torch.as_tensor(q_idx, dtype=torch.long) // block_size
    kv_t = torch.as_tensor(kv_idx, dtype=torch.long) // block_size
    causal = q_t >= kv_t
    win_ok = (q_t - kv_t) < w
    return causal & win_ok


def _block_fast_to_slow(
    b, h, q_idx, kv_idx, block_size: int, rate: int, *, window: Optional[int] = None
) -> torch.Tensor:
    w = _effective_window(window)
    q_t = torch.as_tensor(q_idx, dtype=torch.long) // block_size
    kv_t = torch.as_tensor(kv_idx, dtype=torch.long) // block_size
    f_t, s_t = q_t, kv_t
    base_ok = f_t >= ((s_t + 1) * rate - 1)
    s_max = ((f_t + 1) // rate) - 1
    win_ok = (s_max - s_t) < w
    return base_ok & win_ok


def _block_slow_to_fast(
    b, h, q_idx, kv_idx, block_size: int, rate: int, *, window: Optional[int] = None
) -> torch.Tensor:
    w = _effective_window(window)
    q_t = torch.as_tensor(q_idx, dtype=torch.long) // block_size
    kv_t = torch.as_tensor(kv_idx, dtype=torch.long) // block_size
    s_t, f_t = q_t, kv_t
    base_ok = f_t < ((s_t + 1) * rate)
    win_ok = ((s_t + 1) * rate - 1 - f_t) < w
    return base_ok & win_ok


def _block_self_shifted(
    b, h, q_idx, kv_idx, *, q_shift: int, block_size: int, window: Optional[int]
) -> torch.Tensor:
    return _block_self(b, h, q_idx + q_shift, kv_idx, block_size, window)


def _block_fast_to_slow_shifted(
    b, h, q_idx, kv_idx, *, q_shift: int, block_size: int, rate: int, window: Optional[int]
) -> torch.Tensor:
    return _block_fast_to_slow(
        b, h, q_idx + q_shift, kv_idx, block_size, rate, window=window
    )


def build_self_attn_mask(
    time_len: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_block_mask: bool = False,
    window: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a causal self-attention mask for a sequence with spatiotemporal tokens.

    Token layout is time-major: L = time_len * spatial_tokens.
    For each time step t, all spatial tokens attend to:
      - all spatial tokens at times t' <= t
      - no tokens at times t' > t

    If ``window`` is provided, additionally restrict each query to keys at
    time indices ``[t - window + 1, t]`` (sliding-window causal attention,
    counted in time-block units; window=1 is "self only").

    When use_block_mask=True, return a BlockMask that encodes the same rule
    in block space (block_size == spatial_tokens). Otherwise return a dense
    float mask (0 for allowed, -inf for blocked) compatible with SDPA.
    """
    if use_block_mask:
        q_len = time_len * spatial_tokens
        mask_fn = partial(_block_self, block_size=spatial_tokens, window=window)
        return create_block_mask(mask_fn, None, None, q_len, q_len, device=device)
    if dtype is None:
        dtype = torch.get_default_dtype()
    time_causal = torch.tril(torch.ones(time_len, time_len, device=device, dtype=torch.bool))
    if window is not None:
        idx = torch.arange(time_len, device=device)
        time_window = (idx[:, None] - idx[None, :]) < window
        time_mask = time_causal & time_window
    else:
        time_mask = time_causal
    block = torch.ones(spatial_tokens, spatial_tokens, device=device, dtype=torch.bool)
    allow = torch.kron(time_mask, block)
    return _dense_mask_from_allow(allow, dtype=dtype)


def build_fast_to_slow_mask(
    fast_time: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_block_mask: bool = False,
    window: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a causal cross-attention mask for fast queries attending to slow keys.

    Mapping rule (matches cross_attn1):
      - slow step s summarizes fast steps [s*rate, ..., (s+1)*rate-1]
      - fast time t can attend to slow step s only if t >= (s+1)*rate - 1
        (i.e., the slow summary is fully determined by past fast tokens)

    If ``window`` is provided, additionally restrict each fast query to the
    ``window`` most recent allowed slow keys (counted in slow-time-block
    units). Concretely, with s_max(t) = (t+1)//rate - 1 the latest slow
    index a causal f_t may read, the windowed rule keeps only s in
    [s_max(t) - window + 1, s_max(t)].

    The output shape is [L_fast, L_slow] where:
      L_fast = fast_time * spatial_tokens
      L_slow = slow_time * spatial_tokens
    """
    if use_block_mask:
        q_len = fast_time * spatial_tokens
        kv_len = slow_time * spatial_tokens
        mask_fn = partial(
            _block_fast_to_slow,
            spatial_tokens,
            rate,
            window=window,
        )
        return create_block_mask(mask_fn, None, None, q_len, kv_len, device=device)
    if dtype is None:
        dtype = torch.get_default_dtype()
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = fast_t[:, None] >= ((slow_t[None, :] + 1) * rate - 1)
    if window is not None:
        s_max = ((fast_t[:, None] + 1) // rate) - 1
        allow = allow & ((s_max - slow_t[None, :]) < window)
    return _dense_mask_from_allow(allow, dtype=dtype)


def build_fast_to_slow_block_mask_incremental(
    *,
    lf_total: int,
    nf_query: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    window: Optional[int] = None,
):
    """BlockMask for fast→slow cross when queries are only the last ``nf_query`` fast tokens.

    Shift query indices so ``flex_attention`` treats them as suffix positions of length ``lf_total``.
    """
    q_shift = lf_total - nf_query
    mask_fn = partial(
        _block_fast_to_slow_shifted,
        q_shift=q_shift,
        block_size=spatial_tokens,
        rate=rate,
        window=window,
    )
    kv_len = slow_time * spatial_tokens
    return create_block_mask(mask_fn, None, None, nf_query, kv_len, device=device)


def build_fast_self_block_mask_incremental(
    *,
    lf_total: int,
    nf_query: int,
    spatial_tokens: int,
    device: torch.device,
    window: Optional[int] = None,
):
    """BlockMask for fast self-attention when queries are only the last ``nf_query`` fast tokens.

    ``KVCache`` concatenates past keys with new ones, so attention runs with Q length ``nf_query``
    but K/V length ``lf_total``. Shift query indices so ``mask_mod`` matches global fast layout.
    """
    q_shift = lf_total - nf_query
    mask_fn = partial(
        _block_self_shifted,
        q_shift=q_shift,
        block_size=spatial_tokens,
        window=window,
    )
    return create_block_mask(mask_fn, None, None, nf_query, lf_total, device=device)


def build_slow_to_fast_mask(
    fast_time: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_block_mask: bool = False,
    window: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a causal cross-attention mask for slow queries attending to fast keys.

    Mapping rule (matches cross_attn2):
      - slow step s represents fast range [s*rate, ..., (s+1)*rate-1]
      - slow step s can attend to fast time t iff t < (s+1)*rate, i.e. only
        fast tokens whose index is at most the latest fast index slow_s
        could plausibly summarize.

    If ``window`` is provided, additionally restrict each slow query to the
    ``window`` most recent allowed fast keys. With t_max(s) = (s+1)*rate - 1
    the latest fast index a causal s_s may read, the windowed rule keeps
    only t in [t_max(s) - window + 1, t_max(s)]. With window=rate, slow_s
    attends only to the rate fast tokens it natively summarizes.

    Causality guarantee. Combined with build_fast_to_slow_mask's rule
    (fast_t reads slow_s only when t >= (s+1)*rate - 1), every fast
    output position depends only on fast positions with the same or
    earlier fast index, so future information cannot reach past
    outputs through the slow stream. Without this upper bound the
    previous formulation (`t >= s*rate`) let slow_0 attend to ALL fast
    tokens; that future information then leaked back into past fast
    outputs through subsequent layers, and was the cause of the train
    vs. autoregressive-eval gap that grew with t_num.

    The output shape is [L_slow, L_fast] where:
      L_slow = slow_time * spatial_tokens
      L_fast = fast_time * spatial_tokens
    """
    if use_block_mask:
        q_len = slow_time * spatial_tokens
        kv_len = fast_time * spatial_tokens
        mask_fn = partial(
            _block_slow_to_fast,
            spatial_tokens,
            rate,
            window=window,
        )
        return create_block_mask(mask_fn, None, None, q_len, kv_len, device=device)
    if dtype is None:
        dtype = torch.get_default_dtype()
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = fast_t[None, :] < ((slow_t[:, None] + 1) * rate)
    if window is not None:
        t_max = (slow_t[:, None] + 1) * rate - 1
        allow = allow & ((t_max - fast_t[None, :]) < window)
    return _dense_mask_from_allow(allow, dtype=dtype)


def _dense_mask_from_allow(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """SDPA additive mask: True -> 0, False -> -inf."""
    return torch.zeros_like(allow, dtype=dtype).masked_fill(~allow, float("-inf"))


# -----------------------------------------------------------------------------
# LRU cache for flex BlockMask tuples (disabled). To enable: add ``lru_cache`` to
# the ``functools`` import above, uncomment the constant + function below, and in
# ``_build_masks`` replace the four ``build_*_mask`` calls with a single
# ``_cached_flex_four_block_masks(...)`` unpack (see comment there).
# -----------------------------------------------------------------------------
# from functools import lru_cache, partial  # replace functools import line
#
# _FLEX_BLOCK_MASK_CACHE_MAX = 64
#
# @lru_cache(maxsize=_FLEX_BLOCK_MASK_CACHE_MAX)
# def _cached_flex_four_block_masks(
#     fast_time: int,
#     slow_time: int,
#     device_str: str,
#     spatial_tokens: int,
#     rate: int,
#     self_window: Optional[int],
#     fast_to_slow_window: Optional[int],
#     slow_to_fast_window: Optional[int],
# ) -> tuple:
#     device = torch.device(device_str)
#     fast_self_mask = build_self_attn_mask(
#         fast_time, spatial_tokens, device=device, use_block_mask=True, window=self_window,
#     )
#     slow_self_mask = build_self_attn_mask(
#         slow_time, spatial_tokens, device=device, use_block_mask=True, window=self_window,
#     )
#     fast_to_slow_mask = build_fast_to_slow_mask(
#         fast_time, slow_time, rate, spatial_tokens, device=device,
#         use_block_mask=True, window=fast_to_slow_window,
#     )
#     slow_to_fast_mask = build_slow_to_fast_mask(
#         fast_time, slow_time, rate, spatial_tokens, device=device,
#         use_block_mask=True, window=slow_to_fast_window,
#     )
#     return (fast_self_mask, slow_self_mask, fast_to_slow_mask, slow_to_fast_mask)


logger = getLogger()


class MultiscaleBCAT(nn.Module):
    """
    Wrapper for the autoregressive BCAT model.
    During generation, runs one autoregressive fast step per iteration (same masks
    and pool semantics as ``fwd`` on the current prefix).
    """

    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.rate = config.get("rate", 1)
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)

        if config.get("kv_cache", False) and config.get("rotary", False):
            raise ValueError("MultiscaleBCAT kv_cache rollout requires rotary=0.")

        if config.get("kv_cache", False) and self.flex_attn:
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = 128

        # Optional sliding-window restriction on top of the causal masks. When
        # limit_window=False (default), all windows are None -> pure causal.
        # All window sizes are counted in time-block units (one "step" per
        # group of spatial_tokens).
        if config.get("limit_window", False):
            self._self_window = config.get("self_window", 3)
            self._fast_to_slow_window = config.get("fast_to_slow_window", 2)
            self._slow_to_fast_window = config.get("slow_to_fast_window", self.rate)
        else:
            self._self_window = None
            self._fast_to_slow_window = None
            self._slow_to_fast_window = None

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case "dyt":
                norm = DynamicTanh
            case _:
                norm = nn.LayerNorm

        fast_embed_dim = config.dim_emb
        embedder_dim = config.embedder.dim
        slow_embed_dim = config.get("slow_dim", config.dim_emb)
        #recombine_hidden_dim = config.get("recombine_dim", config.dim_ffn)
        #lift_hidden_dim = config.get("lift_dim", config.dim_ffn)
        pool_hidden_dim = config.get("pool_dim", config.dim_ffn)
        activation = config.get("activation", "gelu")
        #recombine_activation = config.get("recombine_activation", activation)

        pool_ffn = PoolFFN(
            in_dim=embedder_dim,
            out_dim=slow_embed_dim,
            hidden_dim=pool_hidden_dim,
            rate=self.rate,
            act=activation,
            dropout=config.dropout,
        )

        encoder_layer = TwoScaleTransformerEncoderLayer(
            fast_embed_dim=fast_embed_dim,
            slow_embed_dim=slow_embed_dim,
            num_heads=config.n_head,
            rate=self.rate,
            dim_ffn=config.dim_ffn,
            dropout=config.dropout,
            act=activation,
            bias=True,
            qk_norm=config.get("qk_norm", False),
            flex_attn=self.flex_attn,
            flex_contiguous_cached=bool(self.config.get("kv_cache", False)) and self.flex_attn,
            shared_scale_ffn=config.get("shared_scale_ffn", False),
            norm=norm,
        )
        split_encoder = SplitEncoder(
            embedder=self.embedder,
            rate=self.rate,
            pool_ffn=pool_ffn,
            spatial_tokens=config.embedder.patch_num**2,
        )
        decoder = FastOnlyRecombineDecoder(
            fast_embed_dim=fast_embed_dim,
            norm=norm,
            embedder=self.embedder,
        )
        self.transformer = TwoScaleTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layer,
            split_encoder=split_encoder,
            decoder=decoder,
            config=config,
        )

        self.seq_len_per_step = config.embedder.patch_num**2
        # Precompute dense masks for the maximum fast/slow lengths
        self.max_fast_time = max(1, max_data_len - 1)
        self.max_slow_time = max(1, self.max_fast_time // self.rate)
        self.register_buffer(
            "fast_self_mask_full",
            build_self_attn_mask(
                self.max_fast_time,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._self_window,
            ),
            persistent=False,
        )
        self.register_buffer(
            "slow_self_mask_full",
            build_self_attn_mask(
                self.max_slow_time,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._self_window,
            ),
            persistent=False,
        )
        self.register_buffer(
            "fast_to_slow_mask_full",
            build_fast_to_slow_mask(
                self.max_fast_time,
                self.max_slow_time,
                self.rate,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._fast_to_slow_window,
            ),
            persistent=False,
        )
        self.register_buffer(
            "slow_to_fast_mask_full",
            build_slow_to_fast_mask(
                self.max_fast_time,
                self.max_slow_time,
                self.rate,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._slow_to_fast_window,
            ),
            persistent=False,
        )

    def summary(self):
        # The embedder is shared by reference between self.embedder,
        # self.transformer.split_encoder, and self.transformer.decoder
        # (FastOnlyRecombineDecoder). Counting each submodule's
        # `parameters()` independently therefore double- or triple-counts
        # the embedder. We claim each parameter the first time we see it
        # so the per-component numbers are disjoint and sum to the total.
        seen: set[int] = set()

        def _count(module: nn.Module) -> int:
            total = 0
            for p in module.parameters():
                if not p.requires_grad or id(p) in seen:
                    continue
                seen.add(id(p))
                total += p.numel()
            return total

        n_embedder = _count(self.embedder)
        n_pool = _count(self.transformer.split_encoder.pool_ffn)
        n_layers = _count(self.transformer.layers)
        n_decoder = (
            _count(self.transformer.decoder)
            if self.transformer.decoder is not None
            else 0
        )
        n_rotary = (
            _count(self.transformer.rotary_emb)
            if self.transformer.rotary_emb is not None
            else 0
        )
        n_total = n_embedder + n_pool + n_layers + n_decoder + n_rotary

        lines = ["\n", f"\tEmbedder (shared):  {n_embedder:>14,}"]
        lines.append(f"\tPool FFN:           {n_pool:>14,}")
        lines.append(f"\tLayer stack:        {n_layers:>14,}")
        if self.transformer.decoder is not None:
            lines.append(f"\tDecoder:            {n_decoder:>14,}")
        if n_rotary:
            lines.append(f"\tRotary:             {n_rotary:>14,}")
        lines.append(f"\tTotal:   {n_total:>14,}")
        return "\n".join(lines)

    def _build_masks(
        self,
        fast_time: int,
        slow_time: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        use_block_mask = self.flex_attn
        if use_block_mask:
            # Optional LRU: uncomment ``_cached_flex_four_block_masks`` above and use:
            # fast_self_mask, slow_self_mask, fast_to_slow_mask, slow_to_fast_mask = (
            #     _cached_flex_four_block_masks(
            #         fast_time, slow_time, str(device), self.seq_len_per_step, self.rate,
            #         self._self_window, self._fast_to_slow_window, self._slow_to_fast_window,
            #     )
            # )
            fast_self_mask = build_self_attn_mask(
                fast_time,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
                window=self._self_window,
            )
            slow_self_mask = build_self_attn_mask(
                slow_time,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
                window=self._self_window,
            )
            fast_to_slow_mask = build_fast_to_slow_mask(
                fast_time,
                slow_time,
                self.rate,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
                window=self._fast_to_slow_window,
            )
            slow_to_fast_mask = build_slow_to_fast_mask(
                fast_time,
                slow_time,
                self.rate,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
                window=self._slow_to_fast_window,
            )
            return {
                "fast_self_attn_mask": None,
                "slow_self_attn_mask": None,
                "fast_to_slow_attn_mask": None,
                "slow_to_fast_attn_mask": None,
                "fast_block_mask": fast_self_mask,
                "slow_block_mask": slow_self_mask,
                "fast_to_slow_block_mask": fast_to_slow_mask,
                "slow_to_fast_block_mask": slow_to_fast_mask,
            }

        fast_len = fast_time * self.seq_len_per_step
        slow_len = slow_time * self.seq_len_per_step
        return {
            "fast_self_attn_mask": self.fast_self_mask_full[:fast_len, :fast_len].to(dtype=dtype, device=device),
            "slow_self_attn_mask": self.slow_self_mask_full[:slow_len, :slow_len].to(dtype=dtype, device=device),
            "fast_to_slow_attn_mask": self.fast_to_slow_mask_full[:fast_len, :slow_len].to(
                dtype=dtype, device=device
            ),
            "slow_to_fast_attn_mask": self.slow_to_fast_mask_full[:slow_len, :fast_len].to(
                dtype=dtype, device=device
            ),
            "fast_block_mask": None,
            "slow_block_mask": None,
            "fast_to_slow_block_mask": None,
            "slow_to_fast_block_mask": None,
        }

    def _transformer_flex(self, data, times, masks):
        return self.transformer(
            data=data,
            times=times,
            masks=masks,
            is_causal=False,
            spatial_tokens=self.seq_len_per_step,
            full=True,
        )

    def _transformer_dense(self, data, times, masks):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return self.transformer(
                data=data,
                times=times,
                masks=masks,
                is_causal=False,
                spatial_tokens=self.seq_len_per_step,
                full=True,
            )

    # Added by Cursor Composer 2.
    def _kv_rollout_embed_and_slow(
        self,
        cur_data_input: torch.Tensor,
        times_prefix: torch.Tensor,
        prev_len: int,
        xf_accum: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """BCAT-style ``skip_len`` encode, concatenate along fast tokens, ``pool`` to slow (matches ``SplitEncoder``).


        Returns ``(xf_step, xf_new, lf_tot, xs_slow)``:

        - ``xf_step``: ``[B, L_step, embed_dim]`` where ``L_step`` is fast tokens encoded this call only.
        - ``xf_new``: ``[B, lf_tot, embed_dim]`` full fast prefix after concat (``lf_tot`` = all frames so far flattened to tokens).
        - ``xs_slow``: ``[B, L_slow, slow_dim]`` with ``L_slow = slow_time * spatial_tokens`` matching ``pool``.
        """
        # ``skip_len=prev_len`` encodes only new frames once the KV prefix is warm (aligned with BCAT ``generate``).
        xf_step = self.embedder.encode(
            cur_data_input,
            times_prefix,
            skip_len=prev_len,
        )
        xf_new = xf_step if xf_accum is None else torch.cat([xf_accum, xf_step], dim=1)
        lf_tot = xf_new.size(1)
        se = self.transformer.split_encoder
        # Slow stream recomputed each outer autoregressive step - must match pooled full prefix from training.
        xs_slow = pool(
            xf_new,
            rate=self.rate,
            time_dim=1,
            pool_ffn=se.pool_ffn,
            spatial_tokens=self.seq_len_per_step,
        )
        return xf_step, xf_new, lf_tot, xs_slow

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes
        with small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data, times, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        data = data[:, :-1]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)
        times = times[:, :-1]  # (bs/1, t_num-1, 1)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, t_num-1, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = (input_len + output_len - 1) * patch_num * patch_num
        """
        fast_time = data.size(1)
        slow_time = max(1, fast_time // self.rate)

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        masks = self._build_masks(fast_time, slow_time, device=data.device, dtype=data.dtype)
        if self.flex_attn:
            data_encoded = self._transformer_flex(data=data, times=times, masks=masks)
        else:
            data_encoded = self._transformer_dense(data=data, times=times, masks=masks)

        """
        Step 3: Decode data
        """
        data_output = data_encoded[:, input_len - 1 :]
        return data_output

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        t_num = times.size(1)
        output_len = t_num - input_len
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input
        cur_len = input_len
        prev_len = 0
        fast_time = cur_len
        slow_time = max(1, fast_time // self.rate)

        # KV-cached rollout: two-scale analogue of BCAT ``generate`` with ``KVCache`` (see ``forward_rollout_kv_cache``).
        # Added by Cursor Composer 2.
        kv_cache_rollout = bool(self.config.get("kv_cache", False))
        if kv_cache_rollout:
            max_lf = t_num * self.seq_len_per_step
            attn_cache = KVCache(
                2 * self.config.n_layer,
                bs,
                max_lf,
                self.config.n_head,
                self.config.dim_emb // self.config.n_head,
            )

        # Fast embedding prefix for ``pool`` - kept across AR steps so ``slow_full_residual`` matches ``SplitEncoder``.
        xf_accum = None

        for _i in range(output_len):
            cur_data_input = data_all[:, :cur_len]

            masks = self._build_masks(
                fast_time,
                slow_time,
                device=cur_data_input.device,
                dtype=cur_data_input.dtype,
            )

            if kv_cache_rollout:
                xf_step, xf_accum, lf_tot, xs_slow = self._kv_rollout_embed_and_slow(
                    cur_data_input,
                    times[:, :cur_len],
                    prev_len,
                    xf_accum,
                )
                nt = cur_data_input.size(1)
                st = self.seq_len_per_step
                if lf_tot != nt * st:
                    raise RuntimeError(
                        f"KV rollout: lf_tot ({lf_tot}) != nt*st ({nt}*{st}); "
                        "check embed flatten vs spatial_tokens."
                    )
                new_frames = nt - prev_len
                if xf_step.size(1) != new_frames * st:
                    raise RuntimeError(
                        f"KV rollout: xf_step len ({xf_step.size(1)}) != new_frames*st ({new_frames}*{st}); "
                        "check encode skip_len vs prev_len."
                    )
                if self.flex_attn:
                    masks_kv = dict(masks)
                    # Non-trivial alignment: shortened fast queries need shifted ``mask_mod`` (see incremental builder).
                    if xf_step.size(1) < lf_tot:
                        masks_kv["fast_self_block_mask_incremental"] = build_fast_self_block_mask_incremental(
                            lf_total=lf_tot,
                            nf_query=xf_step.size(1),
                            spatial_tokens=self.seq_len_per_step,
                            device=cur_data_input.device,
                            window=self._self_window,
                        )
                    masks_kv["fast_to_slow_block_mask_incremental"] = build_fast_to_slow_block_mask_incremental(
                        lf_total=lf_tot,
                        nf_query=xf_step.size(1),
                        slow_time=slow_time,
                        rate=self.rate,
                        spatial_tokens=self.seq_len_per_step,
                        device=cur_data_input.device,
                        window=self._fast_to_slow_window,
                    )
                    cur_data_encoded = self.transformer.forward_rollout_kv_cache(
                        fast_incremental_residual=xf_step,
                        total_fast_tokens=lf_tot,
                        slow_full_residual=xs_slow,
                        masks=masks_kv,
                        spatial_tokens=self.seq_len_per_step,
                        kv_cache=attn_cache,
                    )
                else:
                    # CUDNN SDP backend does not combine with flex Attention the way we intend in the flex branch above.
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        cur_data_encoded = self.transformer.forward_rollout_kv_cache(
                            fast_incremental_residual=xf_step,
                            total_fast_tokens=lf_tot,
                            slow_full_residual=xs_slow,
                            masks=masks,
                            spatial_tokens=self.seq_len_per_step,
                            kv_cache=attn_cache,
                        )
            elif self.flex_attn:
                cur_data_encoded = self._transformer_flex(
                    data=cur_data_input,
                    times=times[:, :cur_len],
                    masks=masks,
                )
            else:
                cur_data_encoded = self._transformer_dense(
                    data=cur_data_input,
                    times=times[:, :cur_len],
                    masks=masks,
                )

            new_output = cur_data_encoded[:, -1:]  # (bs, 1, x_num**2*data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1
            fast_time = cur_len
            slow_time = max(1, fast_time // self.rate)

        return data_all[:, input_len:]
