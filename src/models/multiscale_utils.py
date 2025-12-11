import torch
import torch.nn as nn
from typing import Optional
from .attention_utils import get_activation, MultiheadAttention, MultiheadFlexAttention, _get_clones

def lift(s: torch.Tensor, rate: int, time_dim: int = 1, ffn: Optional[nn.Module] = None) -> torch.Tensor:
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
    out = torch.repeat_interleave(out, repeats=rate, dim=time_dim)
    return out

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
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        # x_blocks: [..., rate, C_in]
        last_two = x_blocks.shape[-2:]
        if last_two != (self.rate, self.in_dim):
            raise ValueError(f"Expected trailing shape ({self.rate}, {self.in_dim}), got {last_two}.")
        x_flat = x_blocks.reshape(*x_blocks.shape[:-2], self.rate * self.in_dim)
        y = self.fc2(self.dropout(self.activation(self.fc1(x_flat))))
        return y

def pool(
    f: torch.Tensor,
    rate: int,
    time_dim: int = 1,
    pool_ffn: Optional[nn.Module] = None,
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

    num_blocks = t // rate
    if num_blocks == 0:
        zero = torch.zeros(b, 1, c, dtype=x.dtype, device=x.device)
        return zero.movedim(1, time_dim)

    t_eff = num_blocks * rate
    x_trim = x[:, :t_eff, :]
    x_blocks = x_trim.view(b, num_blocks, rate, c)  # [B, Bk, R, D_fast]

    s_blocks = pool_ffn(x_blocks)  # [B, Bk, D_pool]
    s0 = torch.zeros(b, 1, c, dtype=x.dtype, device=x.device)
    s = torch.cat([s0, s_blocks], dim=1)  # [B, 1+Bk, D_pool]
    s = s[:, :-1, :]  # drop the last time slice to keep T//rate length
    return s.movedim(1, time_dim)

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

    def __init__(self, encoder: nn.Module, rate: int, pool_ffn: nn.Module, time_dim: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        self.rate = rate
        self.pool_ffn = pool_ffn
        self.time_dim = time_dim

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(*args, **kwargs)
        s = pool(f, rate=self.rate, time_dim=self.time_dim, pool_ffn=self.pool_ffn)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_fast: [B, L, fast_embed_dim]
            x_slow: [B, L_slow, slow_embed_dim], with L_slow == L//rate
            where fast_embed_dim = f_d1 + f_d2 and slow_embed_dim = s_d1 + s_d2
        Returns:
            z_fast: [B, L, fast_embed_dim]
            z_slow: [B, L_slow, slow_embed_dim]
        """
        Bf, Lf, Df = x_fast.shape[0], x_fast.shape[1], x_fast.shape[-1]
        Bs, Ls, Ds = x_slow.shape[0], x_slow.shape[1], x_slow.shape[-1]

        # Validate batch, feature dims, and temporal relation
        if Bf != Bs:
            raise ValueError(f"Batch sizes must match: fast {Bf} vs slow {Bs}.")
        if Df != self.fast_embed_dim:
            raise ValueError(f"Fast feature dim must equal fast_embed_dim={self.fast_embed_dim}: got {Df}.")
        if Ds != self.slow_embed_dim:
            raise ValueError(f"Slow feature dim must equal slow_embed_dim={self.slow_embed_dim}: got {Ds}.")
        if Lf // self.rate != Ls:
            raise ValueError(f"Temporal sizes mismatch: L_fast={Lf}, rate={self.rate}, but L_slow={Ls}.")

        # Fast layer:
        y_fast = lift(x_slow, rate=self.rate, time_dim=time_dim, ffn=self.lift_ffn)  # [B, L, D_lift]
        if y_fast.size(self.split_dim) != self.f_d2:
            raise ValueError(f"lift() output dim {y_fast.size(self.split_dim)} must equal fast cross dim {self.f_d2}.")
        # MixedSplit on fast:
        # x_fast: [B, L, f_d1+f_d2], y_fast: [B, L, f_d2] -> z_fast: [B, L, f_d1+f_d2]
        z_fast = self.fast_mixer(
            x=x_fast,
            y=y_fast,
            x_key_padding_mask=None,
            y_key_padding_mask=None,
            attn_mask=None,
            block_mask=None,
            is_causal=False,
            rotary_emb=None,
        )

        # Slow layer:
        y_slow = pool(x_fast, rate=self.rate, time_dim=time_dim, pool_ffn=self.pool_ffn)  # [B, L//rate, D_pool]
        if y_slow.size(self.split_dim) != self.s_d2:
            raise ValueError(f"pool() output dim {y_slow.size(self.split_dim)} must equal slow cross dim {self.s_d2}.")
        # MixedSplit on slow:
        # x_slow: [B, L//rate, s_d1+s_d2], y_slow: [B, L//rate, s_d2] -> z_slow: [B, L//rate, s_d1+s_d2]
        z_slow = self.slow_mixer(
            x=x_slow,
            y=y_slow,
            x_key_padding_mask=None,
            y_key_padding_mask=None,
            attn_mask=None,
            block_mask=None,
            is_causal=False,
            rotary_emb=None,
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
    ) -> None:
        super().__init__()
        self.rate = rate
        self.time_dim = time_dim
        in_dim = fast_embed_dim + slow_embed_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, fast_embed_dim)

    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> torch.Tensor:
        # Align time scales by lifting z_slow to the fast length (no FFN)
        z_slow_lift = lift(z_slow, rate=self.rate, time_dim=self.time_dim)  # [..., L_fast, slow_embed_dim]
        if z_slow_lift.size(self.time_dim) != z_fast.size(self.time_dim):
            raise ValueError("RecombineDecoder: time dimensions do not match after lift().")
        x = torch.cat([z_fast, z_slow_lift], dim=-1)
        y = self.fc2(self.dropout(self.activation(self.fc1(x))))
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
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm_fast = norm_fast
        self.norm_slow = norm_slow
        self.split_encoder = split_encoder
        self.recombine_decoder = recombine_decoder

    def forward(self, *args, time_dim: int = 1, **kwargs):
        """
        Behavior:
          - If split_encoder and recombine_decoder are provided:
              - If called with (x_fast, x_slow), use them directly.
              - Else, call split_encoder(*args, **kwargs) to get (x_fast, x_slow).
              - Run stacked layers and then recombine to produce y: [B, L, fast_embed_dim].
          - Otherwise:
              - Expect (x_fast, x_slow) and return (z_fast, z_slow).
        """
        if self.split_encoder is not None and self.recombine_decoder is not None:
            # Single-stream pipeline
            # Determine input format
            if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
                x_fast, x_slow = args[0], args[1]
            else:
                x_fast, x_slow = self.split_encoder(*args, **kwargs)

            out_fast = x_fast
            out_slow = x_slow
            for mod in self.layers:
                out_fast, out_slow = mod(out_fast, out_slow, time_dim=time_dim)

            if self.norm_fast is not None:
                out_fast = self.norm_fast(out_fast)
            if self.norm_slow is not None:
                out_slow = self.norm_slow(out_slow)

            # Recombine to a fast-sized output
            return self.recombine_decoder(out_fast, out_slow)

        # operate on both streams separately
        x_fast, x_slow = args[0], args[1]

        out_fast = x_fast
        out_slow = x_slow
        for mod in self.layers:
            out_fast, out_slow = mod(out_fast, out_slow, time_dim=time_dim)

        if self.norm_fast is not None:
            out_fast = self.norm_fast(out_fast)
        if self.norm_slow is not None:
            out_slow = self.norm_slow(out_slow)

        return out_fast, out_slow

    def forward_full(self, *args, time_dim: int = 1, **kwargs) -> torch.Tensor:
        """
        Full pipeline:
          inputs -> SplitEncoder -> (x_fast, x_slow) -> stacked layers -> (z_fast, z_slow) -> RecombineDecoder -> y
        Returns:
          y: [B, L, fast_embed_dim]
        """
        if self.split_encoder is None or self.recombine_decoder is None:
            raise ValueError("forward_full requires split_encoder and recombine_decoder to be set.")

        x_fast, x_slow = self.split_encoder(*args, **kwargs)
        z_fast, z_slow = self.forward(x_fast, x_slow, time_dim=time_dim)
        y = self.recombine_decoder(z_fast, z_slow)
        return y


if __name__ == "__main__":
    #model = class(cfg1, cfg2)
    #output = model(input1, input2)
    pass