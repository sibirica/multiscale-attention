from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .attention_utils import get_activation, MultiheadAttention, MultiheadFlexAttention, _get_clones

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
    b, t, c = x.shape[0], x.shape[1], x.shape[-1]
    if t % spatial_tokens != 0:
        raise ValueError("lift expects time length divisible by spatial_tokens.")

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
        x_flat = x_blocks.reshape(*x_blocks.shape[:-2], self.rate * self.in_dim)
        if self.fc_gate is None:
            y = self.fc2(self.dropout(self.activation(self.fc1(x_flat))))
        else:
            y = self.fc2(self.dropout(self.activation(self.fc1(x_flat), self.fc_gate(x_flat))))
        return y

def pool(
    f: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    pool_ffn: Optional[nn.Module] = None,
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
             - compute block outputs s_k = FFN(f_{k*rate:(k+1)*rate-1}), k=0..K-1, where K = T_fast // rate.
             - prepend a zero step and drop the last to maintain causality and keep exact length K.
             - shape is [B, K, D_pool] if time_dim=1; time length is T_fast // rate.
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
    s0 = torch.zeros(b, 1, spatial_tokens, s_blocks.size(-1), dtype=x.dtype, device=x.device)
    s = torch.cat([s0, s_blocks], dim=1)  # [B, 1+Bk, S, D_pool]
    s = s[:, :-1, :, :]  # drop the last time slice to keep T//rate length
    s = rearrange(s, "b t s d -> b (t s) d")
    return s.movedim(1, time_dim)


def pad_slow(
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
    dim = time_dim if time_dim >= 0 else x.dim() + time_dim
    time_len = x.size(dim)
    if time_len % spatial_tokens != 0:
        raise ValueError("time length must be divisible by spatial_tokens.")
    temporal_len = time_len // spatial_tokens
    pad_steps = (-temporal_len) % rate
    if pad_steps == 0:
        return x, 0
    pad_len = pad_steps * spatial_tokens
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad_tensor], dim=dim), pad_steps

class SplitProcessor(nn.Module):
    """
    Split a dimension of a tensor into two parts, process each part with its own module,
    and concatenate the results back along the same dimension.
    Split evenly by default, custom split sizes via split_sizes param.
    """

    def __init__(
        self,
        branch1: nn.Module,
        branch2: nn.Module,
        split_dim: int = -1,
        split_sizes: Optional[tuple[int, int]] = None,
        cat_dim: int = -1,
    ) -> None:
        super().__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.split_dim = split_dim
        self.split_sizes = split_sizes

    def forward(self, x: torch.Tensor, kw1: Optional[dict] = None, kw2: Optional[dict] = None) -> torch.Tensor:
        kw1 = kw1 or {}
        kw2 = kw2 or {}
        feature_dim = x.size(self.split_dim)
        if self.split_sizes is None:
            if feature_dim % 2 != 0:
                raise ValueError("Even split requested but split dim size is not divisible by 2.")
            split_sizes = (feature_dim // 2, feature_dim // 2)
        else:
            if sum(self.split_sizes) != feature_dim:
                raise ValueError(
                    f"split_sizes {self.split_sizes} must sum to split dim size {feature_dim}."
                )
            split_sizes = self.split_sizes

        x1, x2 = torch.split(x, split_sizes, dim=self.split_dim)
        y1 = self.branch1(x1, **kw1)
        y2 = self.branch2(x2, **kw2)
        return y1, y2
        #return torch.cat([y1, y2], dim=self.split_dim)

class SplitEncoder(nn.Module):
    """
    Wrapper that first computes a 'fast' tensor via the provided encoder,
    then pools it to a 'slow' tensor using pool(). Returns (fast, slow).
    """

    def __init__(
        self,
        encoder: nn.Module,
        rate: int,
        pool_ffn: nn.Module,
        time_dim: int = 1,
        spatial_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rate = rate
        self.pool_ffn = pool_ffn
        self.time_dim = time_dim
        self.spatial_tokens = spatial_tokens

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.encoder, "encode"):
            f = self.encoder.encode(*args, **kwargs)
        else:
            f = self.encoder(*args, **kwargs)
        f_pool, _ = pad_slow(
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

class TwoLevelTransformerEncoderLayer(nn.Module):
    """
    Two-level encoder layer with 'fast' and 'slow' levels coupled by lift/pool with relative rate `rate`.
    - Fast layer: mixed self- and cross-attention between fast sequence X_f and lift(X_s).
    - Slow layer: mixed self- and cross-attention between slow sequence X_s and pool(X_f).
    Outputs (Z_f, Z_s).
    """

    def __init__(
        self,
        fast_embed_dim: int,
        slow_embed_dim: int,
        num_heads: int,
        rate: int,
        pool_ffn: nn.Module,
        lift_ffn: nn.Module,
        fast_split_sizes: Optional[tuple[int, int]] = None,
        slow_split_sizes: Optional[tuple[int, int]] = None,
        split_dim: int = -1,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        flex_attn: bool = False,
    ) -> None:
        super().__init__()
        self.fast_embed_dim = fast_embed_dim
        self.slow_embed_dim = slow_embed_dim
        self.rate = rate
        self.split_dim = split_dim
        self.pool_ffn = pool_ffn
        self.lift_ffn = lift_ffn

        # Determine split sizes per level (d1=self branch, d2=cross branch)
        if fast_split_sizes is None:
            if fast_embed_dim % 2 != 0:
                raise ValueError("fast_embed_dim must be even when fast_split_sizes is None.")
            f_d1 = f_d2 = fast_embed_dim // 2
        else:
            if sum(fast_split_sizes) != fast_embed_dim:
                raise ValueError("sum(fast_split_sizes) must equal fast_embed_dim.")
            f_d1, f_d2 = fast_split_sizes

        if slow_split_sizes is None:
            if slow_embed_dim % 2 != 0:
                raise ValueError("slow_embed_dim must be even when slow_split_sizes is None.")
            s_d1 = s_d2 = slow_embed_dim // 2
        else:
            if sum(slow_split_sizes) != slow_embed_dim:
                raise ValueError("sum(slow_split_sizes) must equal slow_embed_dim.")
            s_d1, s_d2 = slow_split_sizes

        # Cross-channel dims must match for coupling
        if f_d2 != s_d2:
            raise ValueError(f"Cross-branch dims must match: fast d2={f_d2}, slow d2={s_d2}.")

        self.f_d1, self.f_d2 = f_d1, f_d2
        self.s_d1, self.s_d2 = s_d1, s_d2

        # Fast mixer: operates on fast sequence, cross to lifted slow (feature size must be s_d2 == f_d2)
        self.fast_mixer = MixedSplitAttention(
            embed_dim=fast_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            split_dim=split_dim,
            split_sizes=(f_d1, f_d2),
            flex_attn=flex_attn,
        )
        # Slow mixer: operates on slow sequence, cross to pooled fast (feature size must be f_d2 == s_d2)
        self.slow_mixer = MixedSplitAttention(
            embed_dim=slow_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            split_dim=split_dim,
            split_sizes=(s_d1, s_d2),
            flex_attn=flex_attn,
        )

    def forward(
        self,
        x_fast: torch.Tensor,   # [B, L, D_fast]
        x_slow: torch.Tensor,   # [B, L//rate, D_slow]
        time_dim: int = 1,
        fast_attn_mask=None,
        fast_block_mask=None,
        is_causal: bool = False,
        spatial_tokens: int = 1,
        rotary_emb=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_fast: [B, L, fast_embed_dim]
            x_slow: [B, L_slow, slow_embed_dim], with L_slow == ceil(L//rate)
            where fast_embed_dim = f_d1 + f_d2 and slow_embed_dim = s_d1 + s_d2
        Returns:
            z_fast: [B, L, fast_embed_dim]
            z_slow: [B, L_slow, slow_embed_dim]
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
        expected_slow_time = (fast_time + self.rate - 1) // self.rate
        if slow_time != expected_slow_time:
            raise ValueError(f"Temporal sizes mismatch: L_fast={Lf}, rate={self.rate}, but L_slow={Ls}.")

        # Fast layer:
        y_fast = lift(
            x_slow,
            rate=self.rate,
            time_dim=time_dim,
            ffn=self.lift_ffn,
            spatial_tokens=spatial_tokens,
        )  # [B, L, D_lift]
        # lift() can include padded steps from the slow stream; trim to match fast length.
        if y_fast.size(time_dim) > Lf:
            y_fast = y_fast.narrow(time_dim, 0, Lf)
        if y_fast.size(self.split_dim) != self.f_d2:
            raise ValueError(f"lift() output dim {y_fast.size(self.split_dim)} must equal fast cross dim {self.f_d2}.")
        # MixedSplit on fast:
        # x_fast: [B, L, f_d1+f_d2], y_fast: [B, L, f_d2] -> z_fast: [B, L, f_d1+f_d2]
        z_fast = self.fast_mixer(
            x=x_fast,
            y=y_fast,
            attn_mask=fast_attn_mask,
            block_mask=fast_block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )

        slow_attn_mask = _downsample_attn_mask(fast_attn_mask, rate=self.rate, spatial_tokens=spatial_tokens)
        slow_block_mask = _downsample_block_mask(fast_block_mask, rate=self.rate, spatial_tokens=spatial_tokens)

        # Slow layer:
        x_fast_pool, _ = pad_slow(
            x_fast,
            rate=self.rate,
            time_dim=time_dim,
            spatial_tokens=spatial_tokens,
        )
        y_slow = pool(
            x_fast_pool,
            rate=self.rate,
            time_dim=time_dim,
            pool_ffn=self.pool_ffn,
            spatial_tokens=spatial_tokens,
        )  # [B, L//rate, D_pool]
        if y_slow.size(time_dim) != Ls:
            raise ValueError("Pooled slow length does not match slow sequence length.")
        if y_slow.size(self.split_dim) != self.s_d2:
            raise ValueError(f"pool() output dim {y_slow.size(self.split_dim)} must equal slow cross dim {self.s_d2}.")
        # MixedSplit on slow:
        # x_slow: [B, L//rate, s_d1+s_d2], y_slow: [B, L//rate, s_d2] -> z_slow: [B, L//rate, s_d1+s_d2]
        z_slow = self.slow_mixer(
            x=x_slow,
            y=y_slow,
            attn_mask=slow_attn_mask,
            block_mask=slow_block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        return z_fast, z_slow

class _SelfAttnBranch(nn.Module):
    def __init__(self, mha: nn.Module):
        super().__init__()
        self.mha = mha

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> torch.Tensor:
        return self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )

class _CrossAttnBranch(nn.Module):
    def __init__(self, mha: nn.Module):
        super().__init__()
        self.mha = mha

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> torch.Tensor:
        return self.mha(
            x,
            y,
            y,
            key_padding_mask=y_key_padding_mask,
            attn_mask=attn_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )

class MixedSplitAttention(nn.Module):
    """
    Mixed self- and cross-attention on split features:
      - First half attends to itself (self-attention on x_1).
      - Second half attends to y (cross-attention from x_2 -> y_2).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        split_dim: int = -1,
        split_sizes: Optional[tuple[int, int]] = None,
        flex_attn: bool = False,
    ):
        super().__init__()
        self.split_dim = split_dim
        self.split_sizes = split_sizes
        self.flex_attn = flex_attn

        if split_sizes is None:
            if embed_dim % 2 != 0:
                raise ValueError("embed_dim must be even for equal split when split_sizes is None.")
            d1 = d2 = embed_dim // 2
        else:
            if sum(split_sizes) != embed_dim:
                raise ValueError("sum(split_sizes) must equal embed_dim.")
            d1, d2 = split_sizes

        self.d1 = d1
        self.d2 = d2

        MHA = MultiheadFlexAttention if flex_attn else MultiheadAttention
        self.self_branch = _SelfAttnBranch(MHA(d1, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm))
        self.cross_branch = _CrossAttnBranch(MHA(d2, num_heads=num_heads, dropout=dropout, bias=bias, qk_norm=qk_norm))
        self.split_processor = SplitProcessor(self.self_branch, self.cross_branch, split_dim=split_dim, split_sizes=(d1, d2))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_key_padding_mask=None,
        y_key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> torch.Tensor:
        # Ensure y has the correct feature size for cross branch
        if y.size(self.split_dim) != self.d2:
            raise ValueError(f"y feature dim ({y.size(self.split_dim)}) must equal cross-branch dim ({self.d2}).")

        # Build per-branch kwargs; execute via SplitProcessor
        kw1 = dict(
            key_padding_mask=x_key_padding_mask,
            attn_mask=attn_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        kw2 = dict(
            y=y,
            y_key_padding_mask=y_key_padding_mask,
            attn_mask=attn_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        out = self.split_processor(x, kw1=kw1, kw2=kw2)
        if isinstance(out, tuple) and len(out) == 2:
            y1, y2 = out
            return torch.cat([y1, y2], dim=self.split_dim)
        return out
 
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
        act: str = "gelu",
        dropout: float = 0.0,
        time_dim: int = 1,
        spatial_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.time_dim = time_dim
        self.spatial_tokens = spatial_tokens
        in_dim = fast_embed_dim + slow_embed_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if act.endswith("glu"):
            self.fc_gate = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_gate = None
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, fast_embed_dim)

    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> torch.Tensor:
        # Align time scales by lifting z_slow to the fast length (no FFN needed)
        z_slow_lift = lift(
            z_slow,
            rate=self.rate,
            time_dim=self.time_dim,
            spatial_tokens=self.spatial_tokens,
        )  # [..., L_fast, slow_embed_dim]
        if z_slow_lift.size(self.time_dim) > z_fast.size(self.time_dim):
            z_slow_lift = z_slow_lift.narrow(self.time_dim, 0, z_fast.size(self.time_dim))
        if z_slow_lift.size(self.time_dim) != z_fast.size(self.time_dim):
            raise ValueError("RecombineDecoder: time dimensions do not match after lift().")
        x = torch.cat([z_fast, z_slow_lift], dim=-1)
        if self.fc_gate is None:
            y = self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            y = self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))
        return y

        
class TwoLevelTransformerEncoder(nn.Module):
    """
    Stack of TwoLevelTransformerEncoderLayer layers, mirroring CustomTransformerEncoder but
    operating on coupled fast/slow streams. Applies layers sequentially and returns (fast, slow).
    """

    def __init__(
        self,
        encoder_layer: TwoLevelTransformerEncoderLayer,
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
        time_dim: int = 1,
        fast_attn_mask=None,
        fast_block_mask=None,
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
                time_dim=time_dim,
                fast_attn_mask=fast_attn_mask,
                fast_block_mask=fast_block_mask,
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
            time_dim=time_dim,
            fast_attn_mask=fast_attn_mask,
            fast_block_mask=fast_block_mask,
            is_causal=is_causal,
            spatial_tokens=spatial_tokens,
            rotary_emb=self.rotary_emb if self.rotary else None,
        )

    def forward_full(
        self,
        *args,
        time_dim: int = 1,
        fast_attn_mask=None,
        fast_block_mask=None,
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
            time_dim=time_dim,
            fast_attn_mask=fast_attn_mask,
            fast_block_mask=fast_block_mask,
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
        time_dim: int = 1,
        fast_attn_mask=None,
        fast_block_mask=None,
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
                time_dim=time_dim,
                fast_attn_mask=fast_attn_mask,
                fast_block_mask=fast_block_mask,
                is_causal=is_causal,
                spatial_tokens=spatial_tokens,
                rotary_emb=rotary_emb,
            )

        if self.norm_fast is not None:
            out_fast = self.norm_fast(out_fast)
        if self.norm_slow is not None:
            out_slow = self.norm_slow(out_slow)

        return out_fast, out_slow


def _downsample_mask(
    mask: Optional[torch.Tensor],
    rate: int,
    spatial_tokens: int = 1,
    reduce_op: str = "all",
    float_mode: str = "all_masked",
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, BlockMask):
        dense = mask.to_dense()
    else:
        dense = mask

    # Pad query/key axes to rounded-up lengths
    pad_value = True if dense.dtype == torch.bool else float("-inf")
    dense, _ = pad_slow(dense, rate=rate, time_dim=-1, spatial_tokens=spatial_tokens, pad_value=pad_value)
    dense, _ = pad_slow(dense, rate=rate, time_dim=-2, spatial_tokens=spatial_tokens, pad_value=pad_value)

    if spatial_tokens > 1:
        # dense: [*prefix, L, L], where L = t_len * spatial_tokens
        t_len = dense.shape[-1] // spatial_tokens
        slow_len = t_len // rate
        # Split each attention axis into (slow_len, rate, spatial_tokens).
        # dense: [*prefix, slow_len, rate, spatial_tokens, slow_len, rate, spatial_tokens]
        dense = rearrange(
            dense,
            "... (slow r s) (slow2 r2 s2) -> ... slow r s slow2 r2 s2",
            slow=slow_len,
            r=rate,
            s=spatial_tokens,
            slow2=slow_len,
            r2=rate,
            s2=spatial_tokens,
        )
        # Reduce over the rate dimension on both query and key axes.
        reduce_dims = (-5, -2)
    else:
        # dense: [*prefix, L, L], where L = slow_len * rate
        prefix_shape = dense.shape[:-2]
        slow_len = dense.shape[-1] // rate
        # Split each attention axis into (slow_len, rate) for temporal downsampling.
        # dense: [*prefix, slow_len, rate, slow_len, rate]
        dense = rearrange(
            dense,
            "... (slow r) (slow2 r2) -> ... slow r slow2 r2",
            slow=slow_len,
            r=rate,
            slow2=slow_len,
            r2=rate,
        )
        # Reduce over the rate dimension on both query and key axes.
        reduce_dims = (-3, -1)

    if dense.dtype == torch.bool:
        # Bool masks: reduce by logical all/any to keep block semantics.
        if reduce_op == "all":
            slow = dense.all(dim=reduce_dims)
        elif reduce_op == "any":
            slow = dense.any(dim=reduce_dims)
        else:
            raise ValueError(f"Unsupported reduce_op for bool mask: {reduce_op}")
        if spatial_tokens > 1:
            return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
        return slow

    if float_mode == "all_masked":
        # Float masks: mark a slow cell as masked only if all finer cells are masked.
        masked = torch.isneginf(dense) | (dense <= -1e9)
        all_masked = masked.all(dim=reduce_dims)
        slow = torch.zeros(all_masked.shape, device=dense.device, dtype=dense.dtype)
        slow = slow.masked_fill(all_masked, float("-inf"))
        if spatial_tokens > 1:
            return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
        return slow
    if float_mode == "amax":
        # Float masks: preserve any allowed score within the rate block.
        slow = dense.amax(dim=reduce_dims)
        if spatial_tokens > 1:
            return rearrange(slow, "... t s t2 s2 -> ... (t s) (t2 s2)")
        return slow
    raise ValueError(f"Unsupported float_mode: {float_mode}")


def _downsample_attn_mask(
    mask: Optional[torch.Tensor], rate: int, spatial_tokens: int = 1
) -> Optional[torch.Tensor]:
    return _downsample_mask(
        mask=mask,
        rate=rate,
        spatial_tokens=spatial_tokens,
        reduce_op="all",
        float_mode="all_masked",
    )


def _downsample_block_mask(
    mask: Optional[torch.Tensor], rate: int, spatial_tokens: int = 1
) -> Optional[torch.Tensor]:
    if isinstance(mask, BlockMask):
        return None
    return _downsample_mask(
        mask=mask,
        rate=rate,
        spatial_tokens=spatial_tokens,
        reduce_op="any",
        float_mode="amax",
    )


if __name__ == "__main__":
    from models.bcat import block_lower_triangular_mask

    torch.manual_seed(0)

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

    class IdentityEncoder(nn.Module):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return x

    pool_ffn = PoolFFN(
        in_dim=fast_embed_dim,
        out_dim=slow_embed_dim // 2,
        hidden_dim=16,
        rate=rate,
        act="gelu",
        dropout=0.0,
    )
    pool_ffn_split = PoolFFN(
        in_dim=fast_embed_dim,
        out_dim=slow_embed_dim,
        hidden_dim=16,
        rate=rate,
        act="gelu",
        dropout=0.0,
    )
    lift_ffn = nn.Linear(slow_embed_dim, fast_embed_dim // 2)

    layer = TwoLevelTransformerEncoderLayer(
        fast_embed_dim=fast_embed_dim,
        slow_embed_dim=slow_embed_dim,
        num_heads=num_heads,
        rate=rate,
        pool_ffn=pool_ffn,
        lift_ffn=lift_ffn,
        dropout=0.0,
        bias=True,
        qk_norm=False,
        flex_attn=False,
    )

    split_encoder = SplitEncoder(
        encoder=IdentityEncoder(),
        rate=rate,
        pool_ffn=pool_ffn_split,
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
    )
    encoder = TwoLevelTransformerEncoder(
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

    block_size = spatial_tokens
    block_num = time_len_fast
    attn_mask = block_lower_triangular_mask(block_size, block_num, use_float=True)

    z_fast, z_slow = layer(
        x_fast=x_fast,
        x_slow=x_slow,
        time_dim=1,
        fast_attn_mask=attn_mask,
        fast_block_mask=None,
        is_causal=False,
        spatial_tokens=spatial_tokens,
    )

    out_full = encoder(
        x_fast,
        time_dim=1,
        fast_attn_mask=attn_mask,
        fast_block_mask=None,
        is_causal=False,
        spatial_tokens=spatial_tokens,
        full=True,
    )

    out_fast, out_slow = encoder(
        x_fast,
        x_slow,
        time_dim=1,
        fast_attn_mask=attn_mask,
        fast_block_mask=None,
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

    print("TwoLevelTransformerEncoderLayer z_fast:", z_fast.shape)
    print("TwoLevelTransformerEncoderLayer z_slow:", z_slow.shape)
    print("TwoLevelTransformerEncoder full output:", out_full.shape)
    print("TwoLevelTransformerEncoder out_fast:", out_fast.shape)
    print("TwoLevelTransformerEncoder out_slow:", out_slow.shape)