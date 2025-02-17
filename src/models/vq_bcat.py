"""
Autoregressive VQ-BCAT model. 
"""

from logging import getLogger
from functools import partial

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from einops import rearrange

from .attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    CacheCustomTransformerEncoder,
    CacheCustomTransformerEncoderLayer,
)
from .embedder import VQEmbedder
from .kv_cache import KVCache


logger = getLogger()


def block_lower_triangular_mask(block_size, block_num, use_float=False):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)
    """
    matrix_size = block_size * block_num
    lower_tri_mask = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    blocks = torch.block_diag(*[block for _ in range(block_num)])
    final_mask = torch.logical_or(lower_tri_mask, blocks)

    if use_float:
        return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))
    else:
        return ~final_mask


def block_causal(b, h, q_idx, kv_idx, block_size):
    return (q_idx // block_size) >= (kv_idx // block_size)


class VQBCAT(nn.Module):
    """
    Wrapper for the autoregressive VQ-BCAT model.
    """

    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = VQEmbedder(config.embedder, max_output_dim)
        self.head = nn.Linear(config.dim_emb, self.embedder.codebook_size)  # classification head

        self.flex_attn = config.get("flex_attn", False)

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

        self.seq_len_per_step = config.embedder.patch_num**2
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)

        if self.flex_attn:
            block_size = config.patch_num**2
            seq_len = block_size * (max_data_len - 1)
            self.block_mask = create_block_mask(
                partial(block_causal, block_size=block_size), None, None, seq_len, seq_len
            )
            # seq_len_eval = block_size * 10  # NOTE: hardcoded input length
            # self.block_mask_eval = create_block_mask(
            #     partial(block_causal, block_size=block_size), None, None, seq_len_eval, seq_len_eval
            # )

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate2(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    @torch.compiler.disable()
    def prepare_data_ids(self, data, input_len):
        with torch.no_grad():
            # prepare labels for loss calculation
            quant, ids = self.embedder.data_to_ids(data)  # (bs, input_len + output_len, p, p)
            labels = ids[:, input_len:].detach().clone()  # (bs, output_len, p, p)
        return quant, labels

    def fwd(self, data, times, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:   LongTensor (bs, output_len, patch_num, patch_num, codebook_size)
            labels:        LongTensor (bs, output_len, patch_num, patch_num)  # Used for loss calculation
        """

        """
        Step 1: Prepare quantized input and ids as labels
            (b, t, x, y, c) -> (b, (t-1)*p*p, d)
        """
        # quant (b, input_len + output_len, p, p)
        # labels (b, output_len, p, p)
        quant, labels = self.prepare_data_ids(data, input_len)

        quant = quant[:, :-1]  # ignore last step for autoregressive training
        times = times[:, :-1]  # (bs/1, t_num-1, 1)
        data = self.embedder.add_embeddings(times, input_quant=quant)  # (bs, data_len, d)

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
        data_output = rearrange(
            data_output, "b (t h w) d -> b t h w d", h=self.embedder.patch_num, w=self.embedder.patch_num
        )

        data_output = self.head(data_output)  # (b, output_len, p, p, codebook_size)

        return data_output, labels

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

            NOTE: ignore carry_over_c for now
        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        t_num = times.size(1)
        output_len = t_num - input_len
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(
            bs, t_num, self.embedder.patch_num, self.embedder.patch_num, dtype=torch.long, device=data_input.device
        )

        _, ids = self.embedder.data_to_ids(data_input)
        data_all[:, :input_len] = ids
        cur_len = input_len
        prev_len = 0

        config = self.config
        if config.kv_cache:
            cache = KVCache(
                config.n_layer, data_input.shape[0], self.mask.size(0), config.n_head, config.dim_emb // config.n_head
            )

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num)

            # (bs, cur_len, x_num, x_num) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input = self.embedder.add_embeddings(
                times[:, :cur_len], input_ids=cur_data_input, skip_len=skip_len
            )  # (bs, data_len, d)

            mask = None
            if (not self.config.kv_cache) or i == 0:
                data_len = cur_len * self.seq_len_per_step
                mask = self.mask[:data_len, :data_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, mask, cache=cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = rearrange(new_output, "b (h w) d -> b h w d", h=self.embedder.patch_num)
            new_logits = self.head(new_output)  # (b, p, p, codebook_size)

            # use greedy decoding for now
            new_output = torch.argmax(new_logits, dim=-1)  # (b, p, p)

            data_all[:, cur_len] = new_output
            prev_len = cur_len
            cur_len += 1

        return self.embedder.ids_to_data(data_all[:, input_len:]) * data_mask

    @torch.compiler.disable()
    def generate2(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

            NOTE: ignore carry_over_c for now
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

        config = self.config
        if config.kv_cache:
            cache = KVCache(
                config.n_layer, data_input.shape[0], self.mask.size(0), config.n_head, config.dim_emb // config.n_head
            )

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            quant, _ = self.embedder.data_to_ids(cur_data_input)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input = self.embedder.add_embeddings(
                times[:, :cur_len], input_quant=quant, skip_len=skip_len
            )  # (bs, data_len, d)

            mask = None
            if (not self.config.kv_cache) or i == 0:
                data_len = cur_len * self.seq_len_per_step
                mask = self.mask[:data_len, :data_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, mask, cache=cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = rearrange(new_output, "b (h w) d -> b h w d", h=self.embedder.patch_num)
            new_logits = self.head(new_output)  # (b, p, p, codebook_size)

            # use greedy decoding for now
            new_output = torch.argmax(new_logits, dim=-1)  # (b, p, p)

            new_output = new_output[:, None]  # (b, 1, p, p)
            new_output = self.embedder.ids_to_data(new_output) * data_mask  # (b, t, x, y, c)

            data_all[:, cur_len : (cur_len + 1)] = new_output

            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]
