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
from .bcat import block_causal
from .embedder import get_embedder
from .multiscale_utils import (
    PoolFFN,
    RecombineDecoder,
    SplitEncoder,
    TwoScaleTransformerEncoder,
    TwoScaleTransformerEncoderLayer,
)


def build_self_attn_mask(
    time_len: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    use_block_mask: bool = False,
) -> torch.Tensor:
    """
    Build a causal self-attention mask for a sequence with spatiotemporal tokens.

    Token layout is time-major: L = time_len * spatial_tokens.
    For each time step t, all spatial tokens attend to:
      - all spatial tokens at times t' <= t
      - no tokens at times t' > t

    When use_block_mask=True, return a BlockMask that encodes the same rule
    in block space (block_size == spatial_tokens). Otherwise return a dense
    float mask (0 for allowed, -inf for blocked) compatible with SDPA.
    """
    if use_block_mask:
        q_len = time_len * spatial_tokens
        return create_block_mask(
            partial(block_causal, block_size=spatial_tokens),
            None,
            None,
            q_len,
            q_len,
            device=device,
        )
    time_mask = torch.tril(torch.ones(time_len, time_len, device=device, dtype=torch.bool))
    block = torch.ones(spatial_tokens, spatial_tokens, device=device, dtype=torch.bool)
    allow = torch.kron(time_mask, block)
    return _dense_mask_from_allow(allow, dtype=dtype)


def build_fast_to_slow_mask(
    fast_time: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    use_block_mask: bool = False,
) -> torch.Tensor:
    """
    Build a causal cross-attention mask for fast queries attending to slow keys.

    Mapping rule (matches cross_attn1):
      - slow step s summarizes fast steps [s*rate, ..., (s+1)*rate-1]
      - fast time t can attend to slow step s only if t >= (s+1)*rate - 1
        (i.e., the slow summary is fully determined by past fast tokens)

    The output shape is [L_fast, L_slow] where:
      L_fast = fast_time * spatial_tokens
      L_slow = slow_time * spatial_tokens
    """
    if use_block_mask:
        q_len = fast_time * spatial_tokens
        kv_len = slow_time * spatial_tokens
        return create_block_mask(
            partial(_block_fast_to_slow, block_size=spatial_tokens, rate=rate),
            None,
            None,
            q_len,
            kv_len,
            device=device,
        )
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = (fast_t[:, None] >= ((slow_t[None, :] + 1) * rate - 1))
    return _dense_mask_from_allow(allow, dtype=dtype)


def build_slow_to_fast_mask(
    fast_time: int,
    slow_time: int,
    rate: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    use_block_mask: bool = False,
) -> torch.Tensor:
    """
    Build a causal cross-attention mask for slow queries attending to fast keys.

    Mapping rule (matches cross_attn2):
      - slow step s represents fast range [s*rate, ..., (s+1)*rate-1]
      - slow step s can attend to fast time t if t >= s*rate
        (allow contemporaneous and past fast tokens)

    The output shape is [L_slow, L_fast] where:
      L_slow = slow_time * spatial_tokens
      L_fast = fast_time * spatial_tokens
    """
    if use_block_mask:
        q_len = slow_time * spatial_tokens
        kv_len = fast_time * spatial_tokens
        return create_block_mask(
            partial(_block_slow_to_fast, block_size=spatial_tokens, rate=rate),
            None,
            None,
            q_len,
            kv_len,
            device=device,
        )
    fast_t = torch.arange(fast_time, device=device).repeat_interleave(spatial_tokens)
    slow_t = torch.arange(slow_time, device=device).repeat_interleave(spatial_tokens)
    allow = fast_t[None, :] >= (slow_t[:, None] * rate)
    return _dense_mask_from_allow(allow, dtype=dtype)


def _dense_mask_from_allow(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a boolean allow matrix into an SDPA-compatible mask.
    True -> 0 (keep), False -> -inf (block).
    """
    return torch.zeros_like(allow, dtype=dtype).masked_fill(~allow, float("-inf"))


def _block_fast_to_slow(b, h, q_idx, kv_idx, block_size: int, rate: int) -> torch.Tensor:
    """
    BlockMask callback for fast->slow attention.
    q_idx and kv_idx are token indices; integer divide by block_size to get time index.
    """
    f_t = q_idx // block_size
    s_t = kv_idx // block_size
    return f_t >= ((s_t + 1) * rate - 1)


def _block_slow_to_fast(b, h, q_idx, kv_idx, block_size: int, rate: int) -> torch.Tensor:
    """
    BlockMask callback for slow->fast attention.
    q_idx and kv_idx are token indices; integer divide by block_size to get time index.
    """
    s_t = q_idx // block_size
    f_t = kv_idx // block_size
    return f_t >= (s_t * rate)


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
            in_dim=embedder_dim,
            out_dim=slow_embed_dim,
            hidden_dim=pool_hidden_dim,
            rate=self.rate,
            act=activation,
            dropout=config.dropout,
        )
        lift_ffn = nn.Linear(slow_embed_dim, fast_embed_dim)

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
        )
        split_encoder = SplitEncoder(
            embedder=self.embedder,
            rate=self.rate,
            pool_ffn=pool_ffn,
            spatial_tokens=config.embedder.patch_num**2,
        )
        recombine_decoder = RecombineDecoder(
            fast_embed_dim=fast_embed_dim,
            slow_embed_dim=slow_embed_dim,
            rate=self.rate,
            hidden_dim=recombine_hidden_dim,
            lift_ffn=lift_ffn,
            lift_dim=fast_embed_dim,
            act=activation,
            dropout=config.dropout,
            spatial_tokens=config.embedder.patch_num**2,
            embedder=self.embedder,
        )
        self.transformer = TwoScaleTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layer,
            norm_fast=norm(fast_embed_dim, eps=1e-5) if config.norm_first else None,
            norm_slow=norm(slow_embed_dim, eps=1e-5) if config.norm_first else None,
            split_encoder=split_encoder,
            recombine_decoder=recombine_decoder,
            config=config,
        )

        self.seq_len_per_step = config.embedder.patch_num**2
        # Precompute dense masks for the maximum fast/slow lengths
        self.max_fast_time = max(1, max_data_len - 1)
        self.max_slow_time = (self.max_fast_time - 1) // self.rate + 1
        self.register_buffer(
            "fast_self_mask_full",
            build_self_attn_mask(
                self.max_fast_time,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                dtype=torch.float32,
                use_block_mask=False,
            ),
            persistent=False,
        )
        self.register_buffer(
            "slow_self_mask_full",
            build_self_attn_mask(
                self.max_slow_time,
                self.seq_len_per_step,
                device=torch.device("cpu"),
                dtype=torch.float32,
                use_block_mask=False,
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
                dtype=torch.float32,
                use_block_mask=False,
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
                dtype=torch.float32,
                use_block_mask=False,
            ),
            persistent=False,
        )

    def summary(self):
        s = "\n"
        s += f"\tEncoder:        {sum([p.numel() for p in self.transformer.split_encoder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        if self.transformer.recombine_decoder is not None:
            s += (
                f"\tDecoder:      "
                f"{sum([p.numel() for p in self.transformer.recombine_decoder.parameters() if p.requires_grad]):,}"
            )
        return s

    def _build_masks(
        self,
        fast_time: int,
        slow_time: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        use_block_mask = self.flex_attn
        if use_block_mask:
            fast_self_mask = build_self_attn_mask(
                fast_time, self.seq_len_per_step, device=device, use_block_mask=True
            )
            slow_self_mask = build_self_attn_mask(
                slow_time, self.seq_len_per_step, device=device, use_block_mask=True
            )
            fast_to_slow_mask = build_fast_to_slow_mask(
                fast_time,
                slow_time,
                self.rate,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
            )
            slow_to_fast_mask = build_slow_to_fast_mask(
                fast_time,
                slow_time,
                self.rate,
                self.seq_len_per_step,
                device=device,
                use_block_mask=True,
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
        slow_time = (fast_time - 1) // self.rate + 1

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        masks = self._build_masks(fast_time, slow_time, device=data.device, dtype=data.dtype)
        if self.flex_attn:
            data_encoded = self.transformer(
                data=data,
                times=times,
                masks=masks,
                is_causal=False,
                spatial_tokens=self.seq_len_per_step,
                full=True,
            )
        else:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(
                    data=data,
                    times=times,
                    masks=masks,
                    is_causal=False,
                    spatial_tokens=self.seq_len_per_step,
                    full=True,
                )

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
        # prev_len = 0
        fast_time = cur_len
        slow_time = (fast_time - 1) // self.rate + 1

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            # skip_len = prev_len if self.config.kv_cache else 0

            masks = self._build_masks(
                fast_time,
                slow_time,
                device=cur_data_input.device,
                dtype=cur_data_input.dtype,
            )

            # does this need SDPA kernel line? -> only if flex_attn is false
            cur_data_encoded = self.transformer(
                    data=cur_data_input,
                    times=times[:, :cur_len],
                    masks=masks,
                    is_causal=False,
                    spatial_tokens=self.seq_len_per_step,
                    full=True,
            )

            # if self.config.kv_cache:
            #     cur_data_encoded = self.transformer(cur_data_input, mask, block_mask=block_mask, cache=cache)
            # else:
            #     cur_data_encoded = self.transformer(cur_data_input, mask, block_mask=block_mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -1:]  # (bs, 1, x_num**2*data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            # prev_len = cur_len
            cur_len += 1
            fast_time = cur_len
            slow_time = (fast_time - 1) // self.rate + 1

        return data_all[:, input_len:]
        
        # OLD GENERATION CODE: attempt to do one slow block forward
        # t_num = times.size(1)
        # output_len = t_num - input_len
        # bs, _, x_num, _, data_dim = data_input.size()

        # data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        # data_all[:, :input_len] = data_input
        # cur_len = input_len
        # # NOTE: kv_cache is disabled for now.

        # remaining = output_len
        # # First block aligns to the next multiple of rate (or full rate if already aligned).
        # # Outputs past desired length are not generated.
        # step_len = (-cur_len) % self.rate
        # if step_len == 0:
        #     step_len = self.rate
        # while remaining > 0:
        #     block_len = min(step_len, remaining)
        #     cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

        #     fast_time = cur_len
        #     slow_time = (fast_time - 1) // self.rate + 1
        #     masks = self._build_masks(
        #         fast_time,
        #         slow_time,
        #         device=cur_data_input.device,
        #         dtype=cur_data_input.dtype,
        #     )

        #     # does this need SDPA kernel line? -> only if flex_attn is false
        #     cur_data_encoded = self.transformer(
        #             data=cur_data_input,
        #             times=times[:, :cur_len],
        #             masks=masks,
        #             is_causal=False,
        #             spatial_tokens=self.seq_len_per_step,
        #             full=True,
        #     )

        #     new_output = cur_data_encoded[:, -block_len:]  # (bs, block_len, x_num, x_num, data_dim)
        #     new_output = new_output * data_mask  # (bs, block_len, x_num, x_num, data_dim)

        #     if carry_over_c >= 0:
        #         new_output[:, :, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

        #     data_all[:, cur_len : cur_len + block_len] = new_output
        #     cur_len += block_len
        #     remaining -= block_len
        #     step_len = self.rate

        # return data_all[:, input_len:]
