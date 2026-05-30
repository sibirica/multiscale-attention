"""
This file contains attention layers and related utils.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from torch import Tensor

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding

from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.kv_cache import KVCache
from models.checkpoint_wrapper import checkpoint_wrapper

N_MAX_POSITIONS = 4096  # maximum input sequence length
flex_attn_compiled = torch.compile(flex_attention)

"""
--------------- Attention Variants ---------------
"""


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act: str = "gelu", dropout: float = 0.0, bias: bool = True):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim, bias)

        if act.endswith("glu"):
            self.fc_gate = nn.Linear(dim, hidden_dim, bias)
        else:
            self.fc_gate = None

        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim, bias)

    def forward(self, x: Tensor):
        if self.fc_gate is None:
            return self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            return self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, qk_norm: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.qk_norm = qk_norm
        if qk_norm:
            # norm = nn.RMSNorm
            norm = nn.LayerNorm

            self.q_norm = norm(self.head_dim, eps=1e-5)
            self.k_norm = norm(self.head_dim, eps=1e-5)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
        cache: KVCache | None = None,
    ):
        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (bs, n_head, seq_len, head_dim)
        q = q.transpose(1, 2).contiguous()  # make torch.compile happy, striding error otherwise
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rotary_emb is not None:
            q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)

        if cache is not None:
            k, v = cache.update(k, v)
            k_len = k.size(2)

        # process and merge masks
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bs,
                k_len,
            ), f"expecting key_padding_mask shape of {(bs, k_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, k_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        dropout_p = 0.0 if not self.training else self.dropout

        # with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )  # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)


class MultiheadFlexAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        logit_softcap: int = 0,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, qk_norm)

        if logit_softcap > 0:
            from .softcapping import generate_tanh_softcap

            self.score_mod = generate_tanh_softcap(logit_softcap, approx=True)
        else:
            self.score_mod = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
        cache: KVCache | None = None,
    ):
        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)

        if self.qk_norm:
            dtype = q.dtype  # it seems flexattention doesn't autocast to bfloat16
            q = self.q_norm(q).to(dtype)
            k = self.k_norm(k).to(dtype)

        # (bs, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rotary_emb is not None:
            q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)

        if cache is not None:
            k, v = cache.update(k, v)
            k_len = k.size(2)

        output = flex_attn_compiled(
            q, k, v, block_mask=block_mask, score_mod=self.score_mod
        )  # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom implementation of pytorch's TransformerEncoderLayer
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0,
        attn_dropout: float = 0,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype=None,
        rotary: bool = False,
        norm: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        flex_attn: bool = False,
        logit_softcap: float = 0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerEncoderLayer, self).__init__()

        if flex_attn:
            self.self_attn = MultiheadFlexAttention(
                d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm, logit_softcap=logit_softcap
            )
        else:
            assert logit_softcap <= 0, "logit_softcap only works with flex_attn"
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm)
        self.rotary = rotary

        self.ffn = FFN(d_model, dim_feedforward, activation, dropout, bias)

        self.norm_first = norm_first

        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                block_mask=block_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, block_mask=block_mask, is_causal=is_causal, rotary_emb=rotary_emb
                )
            )
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        return self.dropout1(x)


class CustomTransformerEncoder(nn.Module):
    """
    Custom implementation of pytorch's TransformerEncoder
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        config=None,
    ) -> None:
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        if config is not None and config.rotary:
            self.rotary_emb = RotaryEmbedding(dim=config.dim_emb // config.n_head // 2)
            self.rotary = True
        else:
            self.rotary_emb = None
            self.rotary = False

        if config is not None and config.activation_checkpointing:
            for layer in self.layers:
                checkpoint_wrapper(layer)

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
    ):
        # prepare masks
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                block_mask=block_mask,
                rotary_emb=self.rotary_emb,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CacheCustomTransformerEncoderLayer(CustomTransformerEncoderLayer):
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
        cache: KVCache | None = None,
    ):
        if self.training:
            return super().forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                block_mask=block_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )

        assert rotary_emb is None

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        if src.size(1) == 1:
            is_causal = False

        x = src
        x = x + self._sa_block(
            self.norm1(x),
            src_mask,
            src_key_padding_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
            cache=cache,
        )
        x = x + self.dropout2(self.ffn(self.norm2(x)))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
        cache: KVCache | None = None,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
            cache=cache,
        )
        return self.dropout1(x)


class CacheCustomTransformerEncoder(CustomTransformerEncoder):
    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        block_mask=None,
        is_causal: bool = False,
        cache: KVCache | None = None,
    ):
        if self.training:
            if cache is not None:
                raise ValueError("cache should be None in training mode")

            return super().forward(
                src=src,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                block_mask=block_mask,
                is_causal=is_causal,
            )

        output = src

        for i, mod in enumerate(self.layers):
            cache.set_layer(i)
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                block_mask=block_mask,
                rotary_emb=self.rotary_emb,
                cache=cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        config=None,
    ) -> None:
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        if config is not None and config.rotary:
            self.rotary_emb = RotaryEmbedding(dim=config.dim_emb // config.n_head // 2)
            self.rotary = True
        else:
            self.rotary_emb = None
            self.rotary = False

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class OperatorDecoderLayer(nn.Module):
    """OperatorDecoderLayer is made up of multi-head-attn and feedforward network.
    (It is the usual encoder-decoder attention without the self-attention layers)

    Check https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html for details
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0,
        attn_dropout: float = 0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype=None,
        rotary: bool = False,
        norm: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        flex_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if flex_attn:
            self.multihead_attn = MultiheadFlexAttention(
                d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm
            )
        else:
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm)
        self.rotary = rotary

        self.ffn = FFN(d_model, dim_feedforward, activation, dropout)

        self.norm_first = norm_first

        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(
                self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb
            )
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb
                )
            )
            x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        return self.dropout1(x)


"""
--------------- Positional Embeddings ---------------
"""


class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional embedding.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
        """

        if batch_first:
            x = x + self.pe[: x.size(1)].transpose(0, 1)
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LearnablePE(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = Embedding(max_len, d_model)

    def forward(self, x: Tensor, positions: Tensor | None = None, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
            positions: Tensor [batch_size, seq_len]
        """
        seq_len = x.size(1) if batch_first else x.size(0)
        if positions is None:
            positions = x.new(seq_len).long()
            positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)

        pe = self.pe(positions)  # (1, seq_len, d_model)
        if batch_first:
            x = x + pe.expand_as(x)
        else:
            x = x + pe.transpose(0, 1).expand_as(x)

        return self.dropout(x)


def get_embeddings(size: list | tuple, type: str | None = None):
    match type:
        case None:
            patch_embeddings = nn.Parameter(torch.randn(*size))
        case "normalize":
            dim = size[-1]
            patch_embeddings = nn.Parameter((dim**-0.5) * torch.randn(*size))
        case "bert":
            patch_embeddings = nn.Parameter(torch.empty(*size).normal_(std=0.02))
        case _:
            raise ValueError(f"Unknown type for embedding: {type}")
    return patch_embeddings


class GLU(nn.Module):
    def forward(self, x: Tensor, gates: Tensor | None = None) -> Tensor:
        if gates is None:
            x, gates = x.chunk(2, dim=-1)
        return self.act(x) * gates


class GeGLU(GLU):
    def __init__(self):
        super().__init__()
        self.act = nn.GELU()


class SwiGLU(GLU):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()


def get_activation(act: str = "gelu"):
    match act:
        case "relu":
            return nn.ReLU
        case "gelu":
            return nn.GELU
            # return partial(nn.GELU, approximate="tanh")
        case "silu":
            return nn.SiLU
        case "tanh":
            return nn.Tanh
        case "geglu":
            return GeGLU
        case "swiglu":
            return SwiGLU
        case _:
            raise ValueError(f"Unknown activation function: {act}")


"""
--------------- Helper functions ---------------
"""


def get_padding_mask(lengths: torch.LongTensor, max_len: int | None = None):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           int | None         if None, max_len = lengths.max()
    Output:
        key_padding_mask:  BoolTensor (bs, max_len)    (positions with value True are padding)
    """
    if max_len is None:
        max_len = lengths.max().item()

    bs = lengths.size(0)
    key_padding_mask = torch.arange(max_len, device=lengths.device).expand(bs, max_len) >= lengths.unsqueeze(1)
    return key_padding_mask


def get_block_attn_mask(block_size: int, n_repeat: int, device=torch.device("cpu")):
    """
    Output:
        attn_mask: BoolTensor (block_size * n_repeat, block_size * n_repeat) block diagonal matrix with identity blocks
    """
    blocks = [torch.ones(block_size, block_size, device=device)] * n_repeat
    return torch.block_diag(*blocks).bool()


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def Embedding(num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class GroupNorm(nn.Module):
    """
    Channel last Group norm. Expects input of shape (b, t, h, w, c).
    """

    def __init__(self, num_groups: int, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: Tensor):
        b = x.size(0)
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = self.norm(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b)
        return x

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(**self.norm.__dict__)


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value: float = 0.5, **kwargs):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}"


import numbers


class RMSNorm(nn.Module):
    """
    RMSNorm supporting zero-center gamma.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float | None = 1e-5,
        elementwise_affine: bool = True,
        zero_center: bool = True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.elementwise_affine = elementwise_affine
        self.add_one = False
        if self.elementwise_affine:
            if zero_center:
                self.weight = nn.Parameter(torch.zeros(self.normalized_shape))
                self.add_one = True
            else:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight + 1 if self.add_one else self.weight
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class LayerNorm(nn.Module):
    """
    LayerNorm supporting zero-center gamma.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        zero_center: bool = True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.add_one = False
        if self.elementwise_affine:
            if zero_center:
                self.weight = nn.Parameter(torch.zeros(self.normalized_shape))
                self.add_one = True
            else:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight + 1 if self.add_one else self.weight
        return F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)
