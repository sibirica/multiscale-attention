from typing import Optional
import torch
import torch.nn as nn

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .attention_utils import FFN, get_activation, MultiheadAttention, MultiheadFlexAttention, _get_clones

def lift(
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

# TO DO: double check if causality offset by rate or rate-1 (minimum you need)? --> not critical
def pool(
    f: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    pool_ffn: Optional[PoolFFN] = None,
    spatial_tokens: int = 1,
) -> torch.Tensor:
    """
    Causal pooling from fast to slow time scale:
      s_0 = 0
      s_1 = FFN(f_0, ..., f_{rate-1})
      s_2 = FFN(f_{rate}, ..., f_{2*rate-1})
      ...

    Each block of `rate` fast steps maps to one slow step via a block-FFN.

    Args:
        f: fast sequence tensor. Typical shape: [B, T_fast, D_fast] if time_dim=1.
        rate: positive integer; number of fast steps per slow step.
        time_dim: index of the time axis in `f`.
        pool_ffn: required module that maps blocks of shape [..., rate, D_fast] -> [..., D_pool].
                  For example, PoolFFN(in_dim=D_fast, out_dim=D_pool, hidden_dim=..., rate=rate).
    Returns:
        out: pooled slow sequence with causal shift:
             - compute block outputs s_k = FFN(f_{k*rate:(k+1)*rate-1}), k=0..K-1, where K = ceil(T_fast / rate).
             - prepend a zero step and drop the last to maintain causality and keep exact length K.
             - shape is [B, K, D_pool] if time_dim=1; time length is ceil(T_fast / rate).
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
        zero = torch.zeros(b, 1, out_dim, dtype=x.dtype, device=x.device)
        return zero.movedim(1, time_dim)

    if spatial_tokens > 1:
        t_eff = num_blocks * rate
        x_trim = x[:, : t_eff * spatial_tokens, :]
        # [B, T*S, D] -> [B, Bk, S, R, D]
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
        s_blocks = s_blocks.unsqueeze(2) # change first case to [B, Bk, 1, D_pool]
    s0 = torch.zeros(b, 1, spatial_tokens, s_blocks.size(-1), dtype=x.dtype, device=x.device)
    s = torch.cat([s0, s_blocks], dim=1)  # [B, 1+Bk, S, D_pool]
    s = s[:, :-1, :, :]  # drop the last time slice to keep ceil(T/rate) length <--- TO DO: CHECK THIS
    s = rearrange(s, "b t s d -> b (t s) d")
    return s.movedim(1, time_dim)


def pad_for_slow(
    x: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    spatial_tokens: int = 1,
    pad_value: Optional[float] = 0.0,
) -> tuple[torch.Tensor, int]:
    """
    Pad the time axis so pooling by `rate` is exact.
    Padding is appended along `time_dim` and repeated across spatial tokens.
    Returns the padded tensor and the number of temporal steps added.
    """
    dim = time_dim if time_dim >= 0 else x.dim() + time_dim # else: wrap around negative index
    time_len = x.size(dim)
    if time_len % spatial_tokens != 0:
        raise ValueError("'time' length must be divisible by spatial_tokens.")
    temporal_len = time_len // spatial_tokens
    pad_steps = (-temporal_len) % rate # remainder to get to a multiple of rate
    if pad_steps == 0:
        return x, 0
    pad_len = pad_steps * spatial_tokens
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = x.new_full(pad_shape, pad_value) # tensor with shape of x with pad_value
    return torch.cat([x, pad_tensor], dim=dim), pad_steps

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

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.embedder.encode(*args, **kwargs)
        f_pool, _ = pad_for_slow(
            f,
            rate=self.rate,
            time_dim=self.time_dim,
            spatial_tokens=self.spatial_tokens,
        )
        s = pool(
            f_pool,
            rate=self.rate,
            time_dim=self.time_dim,
            pool_ffn=self.pool_ffn,
            spatial_tokens=self.spatial_tokens,
        )
        return f, s

# TO DO (note): is_causal is somewhat redundant since the masks are causal anyway, 
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
        norm: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.fast_embed_dim = fast_embed_dim
        self.slow_embed_dim = slow_embed_dim
        self.rate = rate

        if fast_embed_dim != slow_embed_dim:
            raise ValueError("Cross-attention requires fast_embed_dim == slow_embed_dim.")

        SelfMHA = MultiheadFlexAttention if flex_attn else MultiheadAttention
        CrossMHA = MultiheadFlexAttention if flex_attn else MultiheadAttention
        self.self_attn_fast = SelfMHA(
            fast_embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm
        )
        self.self_attn_slow = SelfMHA(
            slow_embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm
        )
        self.cross_attn_fast = CrossMHA(
            fast_embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm
        )
        self.cross_attn_slow = CrossMHA(
            slow_embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm
        )

        self.norm_fast_attn = norm(fast_embed_dim)
        self.norm_slow_attn = norm(slow_embed_dim)
        self.norm_fast_ffn = norm(fast_embed_dim)
        self.norm_slow_ffn = norm(slow_embed_dim)

        self.ffn_fast = FFN(fast_embed_dim, dim_ffn, act=act, dropout=dropout)
        self.ffn_slow = FFN(slow_embed_dim, dim_ffn, act=act, dropout=dropout)

    def forward(
        self,
        x_fast: torch.Tensor,   # [B, L_fast, D_fast]
        x_slow: torch.Tensor,   # [B, L_slow, D_slow]
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
        expected_slow_time = (fast_time - 1) // self.rate + 1 # ceil(fast_time/rate)
        if slow_time != expected_slow_time:
            raise ValueError(f"Temporal sizes mismatch: L_fast={Lf}, rate={self.rate}, but L_slow={Ls}.")

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
        x_fast = x_f_res + z_f1 + z_f2
        x_slow = x_s_res + z_s1 + z_s2

        # FFN with residual (per stream)
        x_fast = x_fast + self.ffn_fast(self.norm_fast_ffn(x_fast))
        x_slow = x_slow + self.ffn_slow(self.norm_slow_ffn(x_slow))
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
 
class RecombineDecoder(nn.Module):
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
        act: str = "gelu",
        dropout: float = 0.0,
        time_dim: int = 1,
        spatial_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.time_dim = time_dim
        self.spatial_tokens = spatial_tokens
        self.lift_ffn = lift_ffn
        self.embedder = embedder
        if lift_dim is None:
            if lift_ffn is None:
                lift_dim = slow_embed_dim
            elif hasattr(lift_ffn, "out_features"):
                lift_dim = lift_ffn.out_features
            elif hasattr(lift_ffn, "out_dim"):
                lift_dim = lift_ffn.out_dim
        in_dim = fast_embed_dim + lift_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if act.endswith("glu"):
            self.fc_gate = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_gate = None
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, fast_embed_dim)
        #self.norm_fast = nn.LayerNorm(fast_embed_dim)
        self.norm_fast = nn.LayerNorm(in_dim)

    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> torch.Tensor:
        # Align time scales by lifting z_slow to the fast length
        z_slow_lift = lift(
            z_slow,
            rate=self.rate,
            time_dim=self.time_dim,
            ffn=self.lift_ffn,
            spatial_tokens=self.spatial_tokens,
        )  # [..., L_fast, slow_embed_dim]
        if z_slow_lift.size(self.time_dim) > z_fast.size(self.time_dim):
            z_slow_lift = z_slow_lift.narrow(self.time_dim, 0, z_fast.size(self.time_dim))
        if z_slow_lift.size(self.time_dim) != z_fast.size(self.time_dim):
            raise ValueError("RecombineDecoder: time dimensions do not match after lift().")
        x = torch.cat([z_fast, z_slow_lift], dim=-1)
        #z_fast_res = self.norm_fast(z_fast)
        x = self.norm_fast(x)
        if self.fc_gate is None:
            y = z_fast + self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            y = z_fast + self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))
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
        norm_fast: Optional[nn.Module] = None,
        norm_slow: Optional[nn.Module] = None,
        split_encoder: Optional[SplitEncoder] = None,
        recombine_decoder: Optional[RecombineDecoder] = None,
        config=None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.rate = encoder_layer.rate
        self.norm_fast = norm_fast
        self.norm_slow = norm_slow
        self.split_encoder = split_encoder
        self.recombine_decoder = recombine_decoder
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
        if self.split_encoder is None or self.recombine_decoder is None:
            raise ValueError("forward_full requires split_encoder and recombine_decoder to be set.")

        x_fast, x_slow = self.split_encoder(*args, **kwargs)
        z_fast, z_slow = self._apply_layers(
            x_fast,
            x_slow,
            masks=masks,
            is_causal=is_causal,
            spatial_tokens=spatial_tokens,
            rotary_emb=rotary_emb,
        )
        y = self.recombine_decoder(z_fast, z_slow)
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

        # TO DO: are these needed?
        if self.norm_fast is not None:
            out_fast = self.norm_fast(out_fast)
        if self.norm_slow is not None:
            out_slow = self.norm_slow(out_slow)

        return out_fast, out_slow


# class SplitProcessor(nn.Module):
#     """
#     Split a dimension of a tensor into two parts, process each part with its own module,
#     and concatenate the results back along the same dimension.
#     Split evenly by default, custom split sizes via split_sizes param.
#     """

#     def __init__(
#         self,
#         branch1: nn.Module,
#         branch2: nn.Module,
#         split_dim: int = -1,
#         split_sizes: Optional[tuple[int, int]] = None,
#         cat_dim: int = -1,
#     ) -> None:
#         super().__init__()
#         self.branch1 = branch1
#         self.branch2 = branch2
#         self.split_dim = split_dim
#         self.split_sizes = split_sizes

#     def forward(self, x: torch.Tensor, kw1: Optional[dict] = None, kw2: Optional[dict] = None) -> torch.Tensor:
#         kw1 = kw1 or {}
#         kw2 = kw2 or {}
#         feature_dim = x.size(self.split_dim)
#         if self.split_sizes is None:
#             if feature_dim % 2 != 0:
#                 raise ValueError("Even split requested but split dim size is not divisible by 2.")
#             split_sizes = (feature_dim // 2, feature_dim // 2)
#         else:
#             if sum(self.split_sizes) != feature_dim:
#                 raise ValueError(
#                     f"split_sizes {self.split_sizes} must sum to split dim size {feature_dim}."
#                 )
#             split_sizes = self.split_sizes

#         x1, x2 = torch.split(x, split_sizes, dim=self.split_dim)
#         y1 = self.branch1(x1, **kw1)
#         y2 = self.branch2(x2, **kw2)
#         return y1, y2
#         #return torch.cat([y1, y2], dim=self.split_dim)

# class _SelfAttnBranch(nn.Module):
#     def __init__(self, mha: nn.Module):
#         super().__init__()
#         self.mha = mha

#     def forward(
#         self,
#         x: torch.Tensor,
#         key_padding_mask=None,
#         attn_mask=None,
#         block_mask=None,
#         is_causal: bool = False,
#         rotary_emb=None,
#     ) -> torch.Tensor:
#         return self.mha(
#             x,
#             x,
#             x,
#             key_padding_mask=key_padding_mask,
#             attn_mask=attn_mask,
#             block_mask=block_mask,
#             is_causal=is_causal,
#             rotary_emb=rotary_emb,
#         )

# class _CrossAttnBranch(nn.Module):
#     def __init__(self, mha: nn.Module):
#         super().__init__()
#         self.mha = mha

#     def forward(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         y_key_padding_mask=None,
#         attn_mask=None,
#         block_mask=None,
#         is_causal: bool = False,
#         rotary_emb=None,
#     ) -> torch.Tensor:
#         return self.mha(
#             x,
#             y,
#             y,
#             key_padding_mask=y_key_padding_mask,
#             attn_mask=attn_mask,
#             block_mask=block_mask,
#             is_causal=is_causal,
#             rotary_emb=rotary_emb,
#         )

# Legacy MixedSplitAttention is currently unused after refactor.
# Keeping a commented version of the previous implementation for reference.
# class MixedSplitAttention(nn.Module):
#     """
#     Mixed self- and cross-attention on split features:
#       - First half attends to itself (self-attention on x_1).
#       - Second half attends to y (cross-attention from x_2 -> y_2).
#     """
#
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         bias: bool = True,
#         qk_norm: bool = False,
#         split_dim: int = -1,
#         split_sizes: Optional[tuple[int, int]] = None,
#         flex_attn: bool = False,
#     ):
#         super().__init__()
#         self.split_dim = split_dim
#         self.split_sizes = split_sizes
#         self.flex_attn = flex_attn
#
#         if split_sizes is None:
#             if embed_dim % 2 != 0:
#                 raise ValueError("embed_dim must be even for equal split when split_sizes is None.")
#             d1 = d2 = embed_dim // 2
#         else:
#             if sum(split_sizes) != embed_dim:
#                 raise ValueError("sum(split_sizes) must equal embed_dim.")
#             d1, d2 = split_sizes
#
#         self.d1 = d1
#         self.d2 = d2
#
#         MHA = MultiheadFlexAttention if flex_attn else MultiheadAttention
#         self.self_branch = _SelfAttnBranch(
#             MHA(d1, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm)
#         )
#         self.cross_branch = _CrossAttnBranch(
#             MHA(d2, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm)
#         )
#         self.split_processor = SplitProcessor(
#             self.self_branch,
#             self.cross_branch,
#             split_dim=split_dim,
#             split_sizes=(d1, d2),
#         )
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         x_key_padding_mask=None,
#         y_key_padding_mask=None,
#         attn_mask=None,
#         block_mask=None,
#         is_causal: bool = False,
#         rotary_emb=None,
#     ) -> torch.Tensor:
#         # Ensure y has the correct feature size for cross branch
#         if y.size(self.split_dim) != self.d2:
#             raise ValueError(
#                 f"y feature dim ({y.size(self.split_dim)}) must equal cross-branch dim ({self.d2})."
#             )
#
#         # Build per-branch kwargs; execute via SplitProcessor
#         kw1 = dict(
#             key_padding_mask=x_key_padding_mask,
#             attn_mask=attn_mask,
#             block_mask=block_mask,
#             is_causal=is_causal,
#             rotary_emb=rotary_emb,
#         )
#         kw2 = dict(
#             y=y,
#             y_key_padding_mask=y_key_padding_mask,
#             attn_mask=attn_mask,
#             block_mask=block_mask,
#             is_causal=is_causal,
#             rotary_emb=rotary_emb,
#         )
#         out = self.split_processor(x, kw1=kw1, kw2=kw2)
#         if isinstance(out, tuple) and len(out) == 2:
#             y1, y2 = out
#             return torch.cat([y1, y2], dim=self.split_dim)
#         return out

# Legacy mask downsampling helpers are currently unused after refactor.
# Keep them commented out for reference until fully removed.
# if False:
#     def _downsample_mask(
#         mask: Optional[torch.Tensor],
#         rate: int,
#         spatial_tokens: int = 1,
#         reduce_op: str = "all",
#         float_mode: str = "all_masked",
#     ) -> Optional[torch.Tensor]:
#         if mask is None:
#             return None
#         if isinstance(mask, BlockMask):
#             dense = mask.to_dense()
#         else:
#             dense = mask
#
#         # Pad query/key axes to rounded-up lengths
#         pad_value = True if dense.dtype == torch.bool else float("-inf")
#         dense, _ = pad_for_slow(dense, rate=rate, time_dim=-1, spatial_tokens=spatial_tokens, pad_value=pad_value)
#         dense, _ = pad_for_slow(dense, rate=rate, time_dim=-2, spatial_tokens=spatial_tokens, pad_value=pad_value)
#
#         if spatial_tokens > 1:
#             # dense: [*prefix, L, L], where L = t_len * spatial_tokens
#             t_len = dense.shape[-1] // spatial_tokens
#             slow_len = t_len // rate
#             # Split each attention axis into (slow_len, rate, spatial_tokens).
#             # dense: [*prefix, slow_len, rate, spatial_tokens, slow_len, rate, spatial_tokens]
#             dense = rearrange(
#                 dense,
#                 "... (slow r s) (slow2 r2 s2) -> ... slow r s slow2 r2 s2",
#                 slow=slow_len,
#                 r=rate,
#                 s=spatial_tokens,
#                 slow2=slow_len,
#                 r2=rate,
#                 s2=spatial_tokens,
#             )
#             # Reduce over the rate dimension on both query and key axes.
#             reduce_dims = (-5, -2)
#         else:
#             # dense: [*prefix, L, L], where L = slow_len * rate
#             prefix_shape = dense.shape[:-2]
#             slow_len = dense.shape[-1] // rate
#             # Split each attention axis into (slow_len, rate) for temporal downsampling.
#             # dense: [*prefix, slow_len, rate, slow_len, rate]
#             dense = rearrange(
#                 dense,
#                 "... (slow r) (slow2 r2) -> ... slow r slow2 r2",
#                 slow=slow_len,
#                 r=rate,
#                 slow2=slow_len,
#                 r2=rate,
#             )
#             # Reduce over the rate dimension on both query and key axes.
#             reduce_dims = (-3, -1)
#
#         if dense.dtype == torch.bool:
#             # Bool masks: reduce by logical all/any to keep block semantics.
#             if reduce_op == "all":
#                 slow = dense.all(dim=reduce_dims)
#             elif reduce_op == "any":
#                 slow = dense.any(dim=reduce_dims)
#             else:
#                 raise ValueError(f"Unsupported reduce_op for bool mask: {reduce_op}")
#             if spatial_tokens > 1:
#                 return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
#             return slow
#
#         if float_mode == "all_masked":
#             # Float masks: mark a slow cell as masked only if all finer cells are masked.
#             masked = torch.isneginf(dense) | (dense <= -1e9)
#             all_masked = masked.all(dim=reduce_dims)
#             slow = torch.zeros(all_masked.shape, device=dense.device, dtype=dense.dtype)
#             slow = slow.masked_fill(all_masked, float("-inf"))
#             if spatial_tokens > 1:
#                 return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
#             return slow
#         if float_mode == "amax":
#             # Float masks: preserve any allowed score within the rate block.
#             slow = dense.amax(dim=reduce_dims)
#             if spatial_tokens > 1:
#                 return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
#             return slow
#         raise ValueError(f"Unsupported float_mode: {float_mode}")
#
#
#     def _downsample_attn_mask(
#         mask: Optional[torch.Tensor], rate: int, spatial_tokens: int = 1
#     ) -> Optional[torch.Tensor]:
#         return _downsample_mask(
#             mask=mask,
#             rate=rate,
#             spatial_tokens=spatial_tokens,
#             reduce_op="all",
#             float_mode="all_masked",
#         )
#
#
#     def _downsample_block_mask(
#         mask: Optional[torch.Tensor], rate: int, spatial_tokens: int = 1
#     ) -> Optional[torch.Tensor]:
#         if isinstance(mask, BlockMask):
#             return None
#         return _downsample_mask(
#             mask=mask,
#             rate=rate,
#             spatial_tokens=spatial_tokens,
#             reduce_op="any",
#             float_mode="amax",
#         )

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
    recombine_decoder = RecombineDecoder(
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
        norm_fast=nn.LayerNorm(fast_embed_dim),
        norm_slow=nn.LayerNorm(slow_embed_dim),
        split_encoder=split_encoder,
        recombine_decoder=recombine_decoder,
    )

    fast_seq_len = time_len_fast * spatial_tokens
    slow_seq_len = time_len_slow * spatial_tokens

    x_fast = torch.randn(batch_size, fast_seq_len, fast_embed_dim)
    x_slow = torch.randn(batch_size, slow_seq_len, slow_embed_dim)

    fast_self_mask = build_self_attn_mask(
        time_len_fast, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype
    )
    slow_self_mask = build_self_attn_mask(
        time_len_slow, spatial_tokens, device=x_fast.device, dtype=x_fast.dtype
    )
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