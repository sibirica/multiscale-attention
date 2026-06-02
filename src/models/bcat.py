"""
Autoregressive BCAT model.
"""

from logging import getLogger
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from .attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    CacheCustomTransformerEncoder,
    CacheCustomTransformerEncoderLayer,
    DynamicTanh,
)
from .embedder import get_embedder
from .kv_cache import KVCache


logger = getLogger()


def block_lower_triangular_mask(block_size: int, block_num: int, use_float: bool = False):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)
    """
    return block_lower_triangular_window_mask(block_size, block_num, use_float=use_float)


def block_lower_triangular_window_mask(
    block_size: int,
    block_num: int,
    use_float: bool = False,
    window: Optional[int] = None,
    sink_tokens: int = 0,
):
    matrix_size = block_size * block_num
    time_mask = torch.tril(torch.ones(block_num, block_num, dtype=torch.bool))
    if window is not None:
        idx = torch.arange(block_num)
        within = (idx[:, None] - idx[None, :]) < window
        time_mask = time_mask & within
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    final_mask = torch.kron(time_mask, block)
    if sink_tokens > 0:
        final_mask[:, : min(sink_tokens, matrix_size)] = True

    if use_float:
        return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))
    else:
        return ~final_mask


def block_causal(b, h, q_idx, kv_idx, block_size):
    return (q_idx // block_size) >= (kv_idx // block_size)


def block_causal_window(
    b,
    h,
    q_idx,
    kv_idx,
    *,
    block_size: int,
    window: int,
    sink_tokens: int = 0,
):
    q_t = q_idx // block_size
    k_t = kv_idx // block_size
    sink_ok = kv_idx < sink_tokens
    return sink_ok | ((q_t >= k_t) & ((q_t - k_t) < window))


def block_causal_with_kv_limit_shifted(
    b,
    h,
    q_idx,
    kv_idx,
    *,
    q_shift: int,
    kv_len: int,
    block_size: int,
    window: Optional[int] = None,
    sink_tokens: int = 0,
):
    if kv_idx >= kv_len:
        return False
    if window is None:
        return (kv_idx < sink_tokens) | block_causal(b, h, q_idx + q_shift, kv_idx, block_size)
    return block_causal_window(
        b,
        h,
        q_idx + q_shift,
        kv_idx,
        block_size=block_size,
        window=window,
        sink_tokens=sink_tokens,
    )


class BCAT(nn.Module):
    """
    Wrapper for the autoregressive BCAT model.
    """

    def __init__(self, config, x_num: int, max_output_dim: int, max_data_len: int = 1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)
        if config.get("limit_window", False):
            self._self_window = int(config.get("self_window", 4))
            if self._self_window <= 0:
                raise ValueError(f"self_window must be positive, got {self._self_window}")
        else:
            self._self_window = None
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

        kwargs = {
            "d_model": config.dim_emb,
            "nhead": config.n_head,
            "dim_feedforward": config.dim_ffn,
            "dropout": config.dropout,
            "attn_dropout": config.get("attn_dropout", 0),
            "activation": config.get("activation", "gelu"),
            "norm_first": config.norm_first,
            "norm": norm,
            "rotary": config.rotary,
            "qk_norm": config.get("qk_norm", False),
            "flex_attn": self.flex_attn,
            "bias": config.get("bias", True),
            "logit_softcap": config.get("logit_softcap", 0),
        }

        if config.kv_cache:
            self.transformer = CacheCustomTransformerEncoder(
                CacheCustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )
        else:
            self.transformer = CustomTransformerEncoder(
                CustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )

        self.seq_len_per_step = config.embedder.patch_num**2
        mask = block_lower_triangular_window_mask(
            self.seq_len_per_step,
            max_data_len,
            use_float=True,
            window=self._self_window,
            sink_tokens=self._attn_sink_tokens,
        )
        self.register_buffer("mask", mask, persistent=False)

        self.return_full_cache = False

        if self.flex_attn:
            block_size = config.patch_num**2
            seq_len = block_size * (max_data_len - 1)
            self.block_size = block_size
            self.block_mask = self._create_causal_block_mask(seq_len, seq_len)
            self.block_mask_prefil = None

    def _block_causal_fn(self):
        if self._self_window is None:
            return partial(block_causal, block_size=self.block_size)
        return partial(
            block_causal_window,
            block_size=self.block_size,
            window=self._self_window,
            sink_tokens=self._attn_sink_tokens,
        )

    def _create_causal_block_mask(self, q_len: int, kv_len: int):
        return create_block_mask(self._block_causal_fn(), None, None, q_len, kv_len)

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def compile(self, params):
        self.fwd = torch.compile(self.fwd)

        if params.eval_only:
            self.generate = torch.compile(self.generate)
            self.return_full_cache = True

    def forward(self, mode: str, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        elif mode == "rollout":
            return self.rollout(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data: torch.Tensor, times: torch.Tensor, input_len: int, **kwargs):
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

        data = self.embedder.encode(data, times)  # (bs, data_len, dim)

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1)
        if self.flex_attn:
            block_mask = self._create_causal_block_mask(data_len, data_len)
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:
            mask = self.mask[:data_len, :data_len]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(data, mask=mask)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """

        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    def setup_cache(self, max_batch_size: int, dtype):
        if self.config.kv_cache:
            self.cache = KVCache(
                self.config.n_layer,
                max_batch_size,
                self.mask.size(0),
                self.config.n_head,
                self.config.dim_emb // self.config.n_head,
                dtype=dtype,
                device=next(self.parameters()).device,
                return_full_cache=self.return_full_cache,
            )

    def clear_cache(self):
        self.cache = None

    def prepare_masks(self, step: int, kv_len: int):
        mask = block_mask = None
        if not self.config.kv_cache:
            # regular full square mask
            if self.flex_attn:
                block_mask = self._create_causal_block_mask(kv_len, kv_len)
            else:
                mask = self.mask[:kv_len, :kv_len]
        elif step == 0:
            # first step prefill mask
            if self.return_full_cache:
                if self.flex_attn:
                    block_mask = create_block_mask(
                        partial(
                            block_causal_with_kv_limit_shifted,
                            q_shift=0,
                            kv_len=kv_len,
                            block_size=self.block_size,
                            window=self._self_window,
                            sink_tokens=self._attn_sink_tokens,
                        ),
                        None,
                        None,
                        kv_len,
                        self.mask.size(0),
                    )
                else:
                    mask = self.mask[:kv_len]
            else:
                if self.flex_attn:
                    block_mask = self._create_causal_block_mask(kv_len, kv_len)
                else:
                    mask = self.mask[:kv_len, :kv_len]
        else:
            # decode mask for kv cache
            if self.return_full_cache:
                if self.flex_attn:
                    block_mask = create_block_mask(
                        partial(
                            block_causal_with_kv_limit_shifted,
                            q_shift=kv_len - self.seq_len_per_step,
                            kv_len=kv_len,
                            block_size=self.block_size,
                            window=self._self_window,
                            sink_tokens=self._attn_sink_tokens,
                        ),
                        None,
                        None,
                        self.seq_len_per_step,
                        self.mask.size(0),
                    )
                else:
                    mask = self.mask[(kv_len - self.seq_len_per_step) : kv_len]

        return mask, block_mask

    def generate(
        self,
        data_input: torch.Tensor,
        times: torch.Tensor,
        input_len: int,
        data_mask: torch.Tensor,
        carry_over_c: int = -1,
        **kwargs,
    ):
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

        if self.config.kv_cache:
            self.cache.reset()

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input = self.embedder.encode(
                cur_data_input, times[:, :cur_len], skip_len=skip_len
            )  # (bs, data_len, dim)

            mask, block_mask = self.prepare_masks(i, kv_len=cur_len * self.seq_len_per_step)

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, mask, block_mask=block_mask, cache=self.cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, mask, block_mask=block_mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = self.embedder.decode(new_output)  # (bs, 1, x_num, x_num, data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]

    def rollout(
        self,
        data_input: torch.Tensor,
        times: torch.Tensor,
        input_len: int,
        data_mask: torch.Tensor,
        normalizer: Callable,
        carry_over_c: int = -1,
        **kwargs,
    ):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        assert input_len == 1
        t_num = times.size(1)
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input

        for i in range(input_len, t_num):
            cur_data_input = data_all[:, i - input_len : i]  # (bs, 1, x_num, x_num, data_dim)
            cur_data_input, _, mean, std = normalizer(cur_data_input)

            cur_data_input = self.embedder.encode(cur_data_input, times[:, :input_len])  # (bs, data_len, dim)

            mask = None
            cur_data_encoded = self.transformer(cur_data_input, mask)  # (bs, data_len, dim)

            new_output = self.embedder.decode(cur_data_encoded)  # (bs, 1, x_num, x_num, data_dim)
            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)
            new_output = new_output * std + mean

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, i : i + 1] = new_output

        return data_all[:, input_len:]


class BCAT_Reg(nn.Module):
    """
    Wrapper for the autoregressive BCAT model with additional registers.
    """

    def __init__(self, config, x_num: int, max_output_dim: int, max_data_len: int = 1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)
        if self.flex_attn:
            raise NotImplementedError("flex_attn is not supported in BCAT_Reg right now")

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case _:
                norm = nn.LayerNorm

        kwargs = {
            "d_model": config.dim_emb,
            "nhead": config.n_head,
            "dim_feedforward": config.dim_ffn,
            "dropout": config.dropout,
            "attn_dropout": config.get("attn_dropout", 0),
            "activation": config.get("activation", "gelu"),
            "norm_first": config.norm_first,
            "norm": norm,
            "rotary": config.rotary,
            "qk_norm": config.get("qk_norm", False),
            "flex_attn": self.flex_attn,
        }

        if config.kv_cache:
            self.transformer = CacheCustomTransformerEncoder(
                CacheCustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )
        else:
            self.transformer = CustomTransformerEncoder(
                CustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )

        self.num_reg_token = config.num_reg_token
        self.data_len_per_step = config.embedder.patch_num**2
        self.seq_len_per_step = self.data_len_per_step + self.num_reg_token
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode: str, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data: torch.Tensor, times: torch.Tensor, input_len: int, **kwargs):
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

        data = self.embedder.encode(data, times)  # (bs, data_len, dim)

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1)
        if self.flex_attn:
            block_mask = self.block_mask
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:
            mask = self.mask[:data_len, :data_len]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(data, mask=mask)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """

        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    def setup_cache(self, max_batch_size: int, dtype):
        if self.config.kv_cache:
            self.cache = KVCache(
                self.config.n_layer,
                max_batch_size,
                self.mask.size(0),
                self.config.n_head,
                self.config.dim_emb // self.config.n_head,
                dtype=dtype,
                device=next(self.parameters()).device,
            )

    def clear_cache(self):
        self.cache = None

    @torch.compiler.disable()
    def generate(
        self,
        data_input: torch.Tensor,
        times: torch.Tensor,
        input_len: int,
        data_mask: torch.Tensor,
        carry_over_c: int = -1,
        **kwargs,
    ):
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

        if self.config.kv_cache:
            self.cache.reset()

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input = self.embedder.encode(
                cur_data_input, times[:, :cur_len], skip_len=skip_len
            )  # (bs, data_len, dim)

            mask = None
            if (not self.config.kv_cache) or i == 0:
                data_len = cur_len * self.seq_len_per_step
                mask = self.mask[:data_len, :data_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, mask, cache=self.cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = self.embedder.decode(new_output)  # (bs, 1, x_num, x_num, data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]
