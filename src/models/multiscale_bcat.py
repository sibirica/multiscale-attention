"""
Autoregressive multiscale BCAT model (block generation).
"""

from logging import getLogger
from functools import partial

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from .attention_utils import DynamicTanh
from .embedder import get_embedder
from .bcat import block_lower_triangular_mask, block_causal
from .multiscale_utils import (
    PoolFFN,
    RecombineDecoder,
    SplitEncoder,
    TwoLevelTransformerEncoder,
    TwoLevelTransformerEncoderLayer,
)


logger = getLogger()


class MultiscaleBCAT(nn.Module):
    """
    Wrapper for the autoregressive BCAT model with block generation.
    During generation, predicts `rate` timesteps per iteration.
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
        recombine_hidden_dim = config.get("recombine_dim", config.dim_ffn)
        pool_hidden_dim = config.get("pool_dim", config.dim_ffn)
        activation = config.get("activation", "gelu")


        pool_ffn = PoolFFN(
            in_dim=fast_embed_dim,
            out_dim=slow_embed_dim // 2,
            hidden_dim=pool_hidden_dim,
            rate=self.rate,
            act=activation,
            dropout=config.dropout,
        )
        pool_ffn_split = PoolFFN(
            in_dim=embedder_dim,
            out_dim=slow_embed_dim,
            hidden_dim=pool_hidden_dim,
            rate=self.rate,
            act=activation,
            dropout=config.dropout,
        )
        lift_ffn = nn.Linear(slow_embed_dim, fast_embed_dim // 2)

        encoder_layer = TwoLevelTransformerEncoderLayer(
            fast_embed_dim=fast_embed_dim,
            slow_embed_dim=slow_embed_dim,
            num_heads=config.n_head,
            rate=self.rate,
            pool_ffn=pool_ffn,
            lift_ffn=lift_ffn,
            dropout=config.dropout,
            bias=True,
            qk_norm=config.get("qk_norm", False),
            flex_attn=self.flex_attn,
        )
        split_encoder = SplitEncoder(
            encoder=self.embedder,
            rate=self.rate,
            pool_ffn=pool_ffn_split,
            time_dim=1,
            spatial_tokens=config.embedder.patch_num**2,
        )
        recombine_decoder = RecombineDecoder(
            fast_embed_dim=fast_embed_dim,
            slow_embed_dim=slow_embed_dim,
            rate=self.rate,
            hidden_dim=recombine_hidden_dim,
            act=activation,
            dropout=config.dropout,
            time_dim=1,
            spatial_tokens=config.embedder.patch_num**2,
        )
        self.transformer = TwoLevelTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layer,
            norm_fast=norm(fast_embed_dim, eps=1e-5) if config.norm_first else None,
            norm_slow=norm(slow_embed_dim, eps=1e-5) if config.norm_first else None,
            split_encoder=split_encoder,
            recombine_decoder=recombine_decoder,
            config=config,
        )

        self.seq_len_per_step = config.embedder.patch_num**2
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)

        if self.flex_attn:
            block_size = config.patch_num**2
            seq_len = block_size * (max_data_len - 1)
            self.block_mask = create_block_mask(
                partial(block_causal, block_size=block_size),
                None, None, seq_len, seq_len
            )
            self.block_size = block_size

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        if self.transformer.recombine_decoder is not None:
            s += (
                f"\tRecombine:      "
                f"{sum([p.numel() for p in self.transformer.recombine_decoder.parameters() if p.requires_grad]):,}"
            )
        return s

    def forward(self, mode, **kwargs):
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
        data_len = data.size(1) * self.seq_len_per_step

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        if self.flex_attn:
            block_mask = create_block_mask(
                partial(block_causal, block_size=self.block_size),
                None, None, data_len, data_len, device=data.device
            )
            data_encoded = self.transformer(
                data=data,
                times=times,
                fast_block_mask=block_mask,
                is_causal=False,
                spatial_tokens=self.seq_len_per_step,
                full=True,
            )
        else:
            mask = self.mask[:data_len, :data_len]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(
                    data=data,
                    times=times,
                    fast_attn_mask=mask,
                    is_causal=False,
                    spatial_tokens=self.seq_len_per_step,
                    full=True,
                )

        """
        Step 3: Decode data
        """
        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)
        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
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
        # NOTE: kv_cache is disabled for now.

        step = self.rate
        for i in range(0, output_len, step):
            block_len = min(step, output_len - i)
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            mask = block_mask = None
            if self.flex_attn:
                data_len = cur_len * self.seq_len_per_step
                block_mask = create_block_mask(
                    partial(block_causal, block_size=self.block_size),
                    None, None, data_len, data_len, device=cur_data_input.device
                )
            else:
                data_len = cur_len * self.seq_len_per_step
                mask = self.mask[:data_len, :data_len]

            if self.flex_attn:
                cur_data_encoded = self.transformer(
                    data=cur_data_input,
                    times=times[:, :cur_len],
                    fast_block_mask=block_mask,
                    is_causal=False,
                    spatial_tokens=self.seq_len_per_step,
                    full=True,
                )
            else:
                cur_data_encoded = self.transformer(
                    data=cur_data_input,
                    times=times[:, :cur_len],
                    fast_attn_mask=mask,
                    is_causal=False,
                    spatial_tokens=self.seq_len_per_step,
                    full=True,
                )

            new_tokens = block_len * self.seq_len_per_step
            new_output = cur_data_encoded[:, -new_tokens:]  # (bs, block_len*patch_num*patch_num, dim)
            new_output = self.embedder.decode(new_output)  # (bs, block_len, x_num, x_num, data_dim)
            new_output = new_output * data_mask  # (bs, block_len, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, :, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + block_len] = new_output
            cur_len += block_len

        return data_all[:, input_len:]
