"""
Autoregressive multiscale BCAT model.
"""

from logging import getLogger
from functools import partial, lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from .attention_utils import DynamicTanh
from .bcat import block_causal
from .embedder import get_embedder
from .multiscale_utils import (
    FastOnlyRecombineDecoder,
    PoolFFN,
    SplitEncoder,
    TwoScaleTransformerEncoder,
    TwoScaleTransformerEncoderLayer,
    pool,
)
from .kv_cache import KVCache

def build_self_attn_mask(
    time_len: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_block_mask: bool = False,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    if use_block_mask:
        q_len = time_len * spatial_tokens
        if window is None:
            mask_fn = partial(block_causal, block_size=spatial_tokens)
        else:
            mask_fn = partial(
                _block_self_window, block_size=spatial_tokens, window=window, sink_tokens=sink_tokens
            )
        return create_block_mask(mask_fn, None, None, q_len, q_len, device=device)
    if dtype is None:
        dtype = torch.get_default_dtype()
    time_mask = torch.tril(torch.ones(time_len, time_len, device=device, dtype=torch.bool))
    if window is not None:
        idx = torch.arange(time_len, device=device)
        within = (idx[:, None] - idx[None, :]) < window
        time_mask = time_mask & within
    block = torch.ones(spatial_tokens, spatial_tokens, device=device, dtype=torch.bool)
    allow = torch.kron(time_mask, block)
    if sink_tokens > 0:
        allow[:, : min(sink_tokens, allow.size(1))] = True
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
    sink_tokens: int = 0,
) -> torch.Tensor:
    if use_block_mask:
        q_len = fast_time * spatial_tokens
        kv_len = slow_time * spatial_tokens
        return create_block_mask(
            partial(
                _block_fast_to_slow,
                block_size=spatial_tokens,
                rate=rate,
                window=window,
                sink_tokens=sink_tokens,
            ),
            None,
            None,
            q_len,
            kv_len,
            device=device,
        )
    if dtype is None:
        dtype = torch.get_default_dtype()
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = fast_t[:, None] >= ((slow_t[None, :] + 1) * rate - 1)
    if window is not None:
        s_max = ((fast_t[:, None] + 1) // rate) - 1
        within = (s_max - slow_t[None, :]) < window
        allow = allow & within
    if sink_tokens > 0:
        allow[:, : min(sink_tokens, allow.size(1))] = True
    return _dense_mask_from_allow(allow, dtype=dtype)


def build_slow_to_fast_mask(
    fast_time: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    use_block_mask: bool = False,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    if use_block_mask:
        q_len = slow_time * spatial_tokens
        kv_len = fast_time * spatial_tokens
        return create_block_mask(
            partial(
                _block_slow_to_fast,
                block_size=spatial_tokens,
                rate=rate,
                window=window,
                sink_tokens=sink_tokens,
            ),
            None,
            None,
            q_len,
            kv_len,
            device=device,
        )
    if dtype is None:
        dtype = torch.get_default_dtype()
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = fast_t[None, :] < ((slow_t[:, None] + 1) * rate)
    if window is not None:
        t_max = (slow_t[:, None] + 1) * rate - 1
        within = (t_max - fast_t[None, :]) < window
        allow = allow & within
    if sink_tokens > 0:
        allow[:, : min(sink_tokens, allow.size(1))] = True
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
    sink_tokens: int = 0,
):
    q_shift = lf_total - nf_query
    mask_fn = partial(
        _block_fast_to_slow_shifted,
        q_shift=q_shift,
        block_size=spatial_tokens,
        rate=rate,
        window=window,
        sink_tokens=sink_tokens,
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
    sink_tokens: int = 0,
):
    q_shift = lf_total - nf_query
    mask_fn = partial(
        _block_self_shifted,
        q_shift=q_shift,
        block_size=spatial_tokens,
        window=window,
        sink_tokens=sink_tokens,
    )
    return create_block_mask(mask_fn, None, None, nf_query, lf_total, device=device)


def _dense_mask_from_allow(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros_like(allow, dtype=dtype).masked_fill(~allow, float("-inf"))


def _block_self_window(
    b, h, q_idx, kv_idx, block_size: int, window: int, sink_tokens: int = 0
) -> torch.Tensor:
    q_t = q_idx // block_size
    k_t = kv_idx // block_size
    sink_ok = kv_idx < sink_tokens
    return sink_ok | ((q_t >= k_t) & ((q_t - k_t) < window))


def _block_fast_to_slow(
    b,
    h,
    q_idx,
    kv_idx,
    block_size: int,
    rate: int,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    f_t = q_idx // block_size
    s_t = kv_idx // block_size
    sink_ok = kv_idx < sink_tokens
    causal = f_t >= ((s_t + 1) * rate - 1)
    if window is None:
        return sink_ok | causal
    s_max = ((f_t + 1) // rate) - 1
    return sink_ok | (causal & ((s_max - s_t) < window))


def _block_slow_to_fast(
    b,
    h,
    q_idx,
    kv_idx,
    block_size: int,
    rate: int,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    s_t = q_idx // block_size
    f_t = kv_idx // block_size
    sink_ok = kv_idx < sink_tokens
    causal = f_t < ((s_t + 1) * rate)
    if window is None:
        return sink_ok | causal
    t_max = (s_t + 1) * rate - 1
    return sink_ok | (causal & ((t_max - f_t) < window))


def _block_self_shifted(
    b,
    h,
    q_idx,
    kv_idx,
    *,
    q_shift: int,
    block_size: int,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    if window is None:
        return (kv_idx < sink_tokens) | block_causal(b, h, q_idx + q_shift, kv_idx, block_size)
    return _block_self_window(b, h, q_idx + q_shift, kv_idx, block_size, window, sink_tokens)


def _block_fast_to_slow_shifted(
    b,
    h,
    q_idx,
    kv_idx,
    *,
    q_shift: int,
    block_size: int,
    rate: int,
    window: Optional[int] = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    return _block_fast_to_slow(b, h, q_idx + q_shift, kv_idx, block_size, rate, window, sink_tokens)


# -----------------------------------------------------------------------------
# LRU caches for flex BlockMask tuples
# -----------------------------------------------------------------------------

_FLEX_BLOCK_MASK_CACHE_MAX = 64
_INCREMENTAL_MASK_CACHE_MAX = 4096

@lru_cache(maxsize=_FLEX_BLOCK_MASK_CACHE_MAX)
def _cached_flex_four_block_masks(
    fast_time: int,
    slow_time: int,
    device_str: str,
    spatial_tokens: int,
    rate: int,
    self_window: Optional[int],
    fast_to_slow_window: Optional[int],
    slow_to_fast_window: Optional[int],
    sink_tokens: int,
) -> tuple:
    device = torch.device(device_str)
    fast_self_mask = build_self_attn_mask(
        fast_time, spatial_tokens, device=device, use_block_mask=True, window=self_window, sink_tokens=sink_tokens,
    )
    slow_self_mask = build_self_attn_mask(
        slow_time, spatial_tokens, device=device, use_block_mask=True, window=self_window, sink_tokens=sink_tokens,
    )
    fast_to_slow_mask = build_fast_to_slow_mask(
        fast_time, slow_time, rate, spatial_tokens, device=device,
        use_block_mask=True, window=fast_to_slow_window, sink_tokens=sink_tokens,
    )
    slow_to_fast_mask = build_slow_to_fast_mask(
        fast_time, slow_time, rate, spatial_tokens, device=device,
        use_block_mask=True, window=slow_to_fast_window, sink_tokens=sink_tokens,
    )
    return (fast_self_mask, slow_self_mask, fast_to_slow_mask, slow_to_fast_mask)

@lru_cache(maxsize=_INCREMENTAL_MASK_CACHE_MAX)
def _cached_fast_self_block_mask_incremental(
    lf_total: int,
    nf_query: int,
    spatial_tokens: int,
    device_str: str,
    window: Optional[int],
    sink_tokens: int,
):
    return build_fast_self_block_mask_incremental(
        lf_total=lf_total, nf_query=nf_query, spatial_tokens=spatial_tokens,
        device=torch.device(device_str), window=window, sink_tokens=sink_tokens
    )

@lru_cache(maxsize=_INCREMENTAL_MASK_CACHE_MAX)
def _cached_fast_to_slow_block_mask_incremental(
    lf_total: int,
    nf_query: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device_str: str,
    window: Optional[int],
    sink_tokens: int,
):
    return build_fast_to_slow_block_mask_incremental(
        lf_total=lf_total, nf_query=nf_query, slow_time=slow_time, rate=rate,
        spatial_tokens=spatial_tokens, device=torch.device(device_str), window=window, sink_tokens=sink_tokens
    )

logger = getLogger()


class MultiscaleBCAT(nn.Module):
    def __init__(self, config, x_num, max_output_dim, max_data_len=1, eval_only=False):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.max_data_len = max_data_len
        self.rate = config.get("rate", 1)
        self.eval_only = eval_only
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)

        if config.get("kv_cache", False) and config.get("rotary", False):
            raise ValueError("MultiscaleBCAT kv_cache rollout requires rotary=0.")

        if config.get("limit_window", False):
            self._self_window = config.get("self_window", 3)
            self._fast_to_slow_window = config.get("fast_to_slow_window", 2)
            self._slow_to_fast_window = config.get("slow_to_fast_window", self.rate)
        else:
            self._self_window = None
            self._fast_to_slow_window = None
            self._slow_to_fast_window = None
        self._attn_sink_tokens = int(config.get("attn_sink_tokens", 0))
        if self._attn_sink_tokens < 0:
            raise ValueError(f"attn_sink_tokens must be >= 0, got {self._attn_sink_tokens}")

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
        pool_hidden_dim = config.get("pool_dim", config.dim_ffn)
        activation = config.get("activation", "gelu")

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
        
        self._compiled_transformer_fwd = self._transformer_flex_uncompiled
        self._compiled_transformer_gen = self._transformer_flex_uncompiled
        self._compiled_kv_rollout = self.transformer.forward_rollout_kv_cache

        self.seq_len_per_step = config.embedder.patch_num**2
        self.max_fast_time = max(1, max_data_len - 1)
        self.max_slow_time = max(1, self.max_fast_time // self.rate)
        self.register_buffer(
            "fast_self_mask_full",
            build_self_attn_mask(
                self.max_fast_time, self.seq_len_per_step, device=torch.device("cpu"),
                use_block_mask=False, window=self._self_window, sink_tokens=self._attn_sink_tokens,
            ),
            persistent=False,
        )
        self.register_buffer(
            "slow_self_mask_full",
            build_self_attn_mask(
                self.max_slow_time, self.seq_len_per_step, device=torch.device("cpu"),
                use_block_mask=False, window=self._self_window, sink_tokens=self._attn_sink_tokens,
            ),
            persistent=False,
        )
        self.register_buffer(
            "fast_to_slow_mask_full",
            build_fast_to_slow_mask(
                self.max_fast_time, self.max_slow_time, self.rate, self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._fast_to_slow_window,
                sink_tokens=self._attn_sink_tokens,
            ),
            persistent=False,
        )
        self.register_buffer(
            "slow_to_fast_mask_full",
            build_slow_to_fast_mask(
                self.max_fast_time, self.max_slow_time, self.rate, self.seq_len_per_step,
                device=torch.device("cpu"),
                use_block_mask=False,
                window=self._slow_to_fast_window,
                sink_tokens=self._attn_sink_tokens,
            ),
            persistent=False,
        )
        self.cache = None
        self.cache_dtype = None

    def summary(self):
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

    def compile(self, params):
        self.fwd = torch.compile(self.fwd)

        if self.flex_attn and (not self.eval_only):
            self._compiled_transformer_fwd = torch.compile(self._transformer_flex_uncompiled)
        if self.flex_attn and params.eval_only:
            self._compiled_transformer_gen = torch.compile(self._transformer_flex_uncompiled)
            self._compiled_kv_rollout = torch.compile(self.transformer.forward_rollout_kv_cache)

        if params.eval_only:
            self.generate = torch.compile(self.generate)

    def _build_masks(
        self,
        fast_time: int,
        slow_time: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        if self.flex_attn:
            fast_self_mask, slow_self_mask, fast_to_slow_mask, slow_to_fast_mask = (
                _cached_flex_four_block_masks(
                    fast_time, slow_time, str(device), self.seq_len_per_step, self.rate,
                    self._self_window,
                    self._fast_to_slow_window,
                    self._slow_to_fast_window,
                    self._attn_sink_tokens,
                )
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

    def _transformer_flex_uncompiled(self, data, times, masks):
        return self.transformer(
            data=data,
            times=times,
            masks=masks,
            is_causal=False,
            spatial_tokens=self.seq_len_per_step,
            full=True,
        )

    def _transformer_flex(self, data, times, masks):
        return self._compiled_transformer_flex(data, times, masks)

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

    def _kv_rollout_embed_and_slow(
        self,
        cur_data_input: torch.Tensor,
        times_prefix: torch.Tensor,
        prev_len: int,
        xf_accum: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        xf_step = self.embedder.encode(
            cur_data_input,
            times_prefix,
            skip_len=prev_len,
        )
        xf_new = xf_step if xf_accum is None else torch.cat([xf_accum, xf_step], dim=1)
        lf_tot = xf_new.size(1)
        se = self.transformer.split_encoder
        xs_slow = pool(
            xf_new,
            rate=self.rate,
            time_dim=1,
            pool_ffn=se.pool_ffn,
            spatial_tokens=self.seq_len_per_step,
        )
        return xf_step, xf_new, lf_tot, xs_slow

    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data, times, input_len: int, **kwargs):
        data = data[:, :-1]
        times = times[:, :-1]

        fast_time = data.size(1)
        slow_time = max(1, fast_time // self.rate)

        masks = self._build_masks(fast_time, slow_time, device=data.device, dtype=data.dtype)
        if self.flex_attn:
            data_encoded = self._compiled_transformer_fwd(data=data, times=times, masks=masks)
        else:
            data_encoded = self._transformer_dense(data=data, times=times, masks=masks)

        data_output = data_encoded[:, input_len - 1 :]
        return data_output

    def setup_cache(self, max_batch_size: int, dtype):
        if self.config.get("kv_cache", False):
            self.cache_dtype = dtype
            max_lf = self.max_data_len * self.seq_len_per_step
            self.cache = KVCache(
                2 * self.config.n_layer,
                max_batch_size,
                max_lf,
                self.config.n_head,
                self.config.dim_emb // self.config.n_head,
                dtype=dtype,
                device=next(self.parameters()).device,
            )
        else:
            self.cache = None
            self.cache_dtype = None

    def clear_cache(self):
        self.cache = None
        self.cache_dtype = None

    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
        t_num = times.size(1)
        output_len = t_num - input_len
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input
        cur_len = input_len
        prev_len = 0
        fast_time = cur_len
        slow_time = max(1, fast_time // self.rate)

        kv_cache_rollout = bool(self.config.get("kv_cache", False))
        if kv_cache_rollout:
            max_lf = t_num * self.seq_len_per_step
            cache_dtype = self.cache_dtype if self.cache_dtype is not None else data_input.dtype
            need_new_cache = self.cache is None
            if not need_new_cache:
                cache_shape = self.cache.k_cache[0].shape
                need_new_cache = (
                    bs > cache_shape[0]
                    or max_lf > cache_shape[2]
                    or self.cache.k_cache[0].dtype != cache_dtype
                    or self.cache.device != data_input.device
                )
            if need_new_cache:
                self.cache = KVCache(
                    2 * self.config.n_layer,
                    bs,
                    max_lf,
                    self.config.n_head,
                    self.config.dim_emb // self.config.n_head,
                    dtype=cache_dtype,
                    device=data_input.device,
                )
                self.cache_dtype = cache_dtype
            self.cache.reset()
            attn_cache = self.cache

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
                    raise RuntimeError(f"KV rollout: lf_tot != nt*st.")
                new_frames = nt - prev_len
                if xf_step.size(1) != new_frames * st:
                    raise RuntimeError(f"KV rollout: xf_step len != new_frames*st.")
                
                if self.flex_attn:
                    masks_kv = dict(masks)
                    if xf_step.size(1) < lf_tot:
                        # Utilizing the new incremental mask LRU Cache
                        masks_kv["fast_self_block_mask_incremental"] = _cached_fast_self_block_mask_incremental(
                            lf_tot,
                            xf_step.size(1),
                            self.seq_len_per_step,
                            str(cur_data_input.device),
                            self._self_window,
                            self._attn_sink_tokens,
                        )
                    # Utilizing the new incremental mask LRU Cache
                    masks_kv["fast_to_slow_block_mask_incremental"] = _cached_fast_to_slow_block_mask_incremental(
                        lf_tot,
                        xf_step.size(1),
                        slow_time,
                        self.rate,
                        self.seq_len_per_step,
                        str(cur_data_input.device),
                        self._fast_to_slow_window,
                        self._attn_sink_tokens,
                    )
                    
                    # Call compiled kv_rollout
                    cur_data_encoded = self._compiled_kv_rollout(
                        fast_incremental_residual=xf_step,
                        total_fast_tokens=lf_tot,
                        slow_full_residual=xs_slow,
                        masks=masks_kv,
                        spatial_tokens=self.seq_len_per_step,
                        kv_cache=attn_cache,
                    )
                else:
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
                cur_data_encoded = self._compiled_transformer_gen(
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

            new_output = cur_data_encoded[:, -1:] # (bs, 1, x_num**2*data_dim)

            new_output = new_output * data_mask # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1
            fast_time = cur_len
            slow_time = max(1, fast_time // self.rate)

        return data_all[:, input_len:]