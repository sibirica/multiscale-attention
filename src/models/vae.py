"""
2D VAE-style patch tokenizer for BCAT.

Based on Hunyuan-Video-1.5 VAE: https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention_utils import get_activation


class HunyuanConv2d(nn.Module):
    """2D convolution with replicate padding, frame-independent port of Hunyuan's causal conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.pad_mode = pad_mode
        # F.pad order for 4D input: (w_left, w_right, h_top, h_bottom)
        self.padding = (kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply replicate-padded 2D convolution."""
        hidden_states = F.pad(hidden_states, self.padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class HunyuanRMSNorm(nn.Module):
    """RMS normalization over the channel dimension for 2D feature maps."""

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x along the channel dimension."""
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16)
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class HunyuanAttnBlock(nn.Module):
    """Full spatial self-attention block for 2D feature maps."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = HunyuanRMSNorm(in_channels)
        self.to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention with a residual connection."""
        identity = x
        x = self.norm(x)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        batch_size, channels, height, width = query.shape
        query = query.reshape(batch_size, channels, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        key = key.reshape(batch_size, channels, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        value = value.reshape(batch_size, channels, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

        x = F.scaled_dot_product_attention(query, key, value)
        x = x.squeeze(1).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + identity


class HunyuanDownsample(nn.Module):
    """DCAE-style 2x spatial downsample via channel packing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        factor = 2 * 2
        self.conv = HunyuanConv2d(in_channels, out_channels // factor, kernel_size=3)
        self.group_size = factor * in_channels // out_channels

    @staticmethod
    def _rearrange(tensor: torch.Tensor, r: int = 2) -> torch.Tensor:
        """Convert (b, c, r*h, r*w) -> (b, r*r*c, h, w)."""
        b, c, packed_h, packed_w = tensor.shape
        h, w = packed_h // r, packed_w // r
        tensor = tensor.view(b, c, h, r, w, r).permute(0, 3, 5, 1, 2, 4)
        return tensor.reshape(b, r * r * c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample x by 2 spatially."""
        h = self._rearrange(self.conv(x))
        shortcut = self._rearrange(x)
        b, _, height, width = shortcut.shape
        shortcut = shortcut.view(b, h.shape[1], self.group_size, height, width).mean(dim=2)
        return h + shortcut


class HunyuanUpsample(nn.Module):
    """DCAE-style 2x spatial upsample via channel unpacking."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        factor = 2 * 2
        self.conv = HunyuanConv2d(in_channels, out_channels * factor, kernel_size=3)
        self.repeats = factor * out_channels // in_channels

    @staticmethod
    def _rearrange(tensor: torch.Tensor, r: int = 2) -> torch.Tensor:
        """Convert (b, r*r*c, h, w) -> (b, c, r*h, r*w)."""
        b, packed_c, h, w = tensor.shape
        c = packed_c // (r * r)
        tensor = tensor.view(b, r, r, c, h, w).permute(0, 3, 4, 1, 5, 2)
        return tensor.reshape(b, c, h * r, w * r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample x by 2 spatially."""
        h = self._rearrange(self.conv(x))
        shortcut = self._rearrange(x.repeat_interleave(repeats=self.repeats, dim=1))
        return h + shortcut


class HunyuanResnetBlock(nn.Module):
    """Residual block with RMSNorm and replicate-padded convolutions."""

    def __init__(self, in_channels: int, out_channels: int | None, act: type[nn.Module]) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = act()
        self.norm1 = HunyuanRMSNorm(in_channels)
        self.conv1 = HunyuanConv2d(in_channels, out_channels, kernel_size=3)
        self.norm2 = HunyuanRMSNorm(out_channels)
        self.conv2 = HunyuanConv2d(out_channels, out_channels, kernel_size=3)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply two norm-act-conv layers with a residual connection."""
        residual = hidden_states

        hidden_states = self.conv1(self.nonlinearity(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.nonlinearity(self.norm2(hidden_states)))

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual


class HunyuanMidBlock(nn.Module):
    """Bottleneck block alternating resnets and spatial attention."""

    def __init__(self, in_channels: int, act: type[nn.Module], num_layers: int = 1) -> None:
        super().__init__()
        resnets = [HunyuanResnetBlock(in_channels, in_channels, act)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(HunyuanAttnBlock(in_channels))
            resnets.append(HunyuanResnetBlock(in_channels, in_channels, act))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck resnet/attention stack."""
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = resnet(attn(hidden_states))
        return hidden_states


class HunyuanDownBlock(nn.Module):
    """Stack of resnets with an optional spatial downsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        act: type[nn.Module],
        downsample_out_channels: int | None,
    ) -> None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(HunyuanResnetBlock(in_channels if i == 0 else out_channels, out_channels, act))
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if downsample_out_channels is not None:
            self.downsamplers = nn.ModuleList([HunyuanDownsample(out_channels, downsample_out_channels)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply resnets then optional downsampling."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class HunyuanUpBlock(nn.Module):
    """Stack of resnets with an optional spatial upsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        act: type[nn.Module],
        upsample_out_channels: int | None,
    ) -> None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(HunyuanResnetBlock(in_channels if i == 0 else out_channels, out_channels, act))
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_out_channels is not None:
            self.upsamplers = nn.ModuleList([HunyuanUpsample(out_channels, upsample_out_channels)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply resnets then optional upsampling."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class HunyuanEncoder2D(nn.Module):
    """Frame-independent 2D encoder ported from Hunyuan's 3D VAE encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        spatial_compression_ratio: int,
        act: type[nn.Module],
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanConv2d(in_channels, block_out_channels[0], kernel_size=3)
        self.down_blocks = nn.ModuleList([])

        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            add_spatial_downsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            if add_spatial_downsample:
                downsample_out_channels = block_out_channels[i + 1]
                self.down_blocks.append(
                    HunyuanDownBlock(input_channel, output_channel, layers_per_block, act, downsample_out_channels)
                )
                input_channel = downsample_out_channels
            else:
                self.down_blocks.append(HunyuanDownBlock(input_channel, output_channel, layers_per_block, act, None))
                input_channel = output_channel

        self.mid_block = HunyuanMidBlock(block_out_channels[-1], act)
        self.norm_out = HunyuanRMSNorm(block_out_channels[-1])
        self.conv_act = act()
        self.conv_out = HunyuanConv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=3)
        # Channel-alignment projection from the deepest feature width to the latent dim.
        self.proj_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode (b, in_channels, h, w) to (b, out_channels, h', w')."""
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_out(self.conv_act(self.norm_out(hidden_states))) + hidden_states
        return self.proj_out(hidden_states)


class HunyuanDecoder2D(nn.Module):
    """Frame-independent 2D decoder ported from Hunyuan's 3D VAE decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        spatial_compression_ratio: int,
        act: type[nn.Module],
    ) -> None:
        super().__init__()

        # Channel-alignment projection from the latent dim to the deepest feature width.
        self.proj_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=1)
        self.conv_in = HunyuanConv2d(block_out_channels[0], block_out_channels[0], kernel_size=3)
        self.mid_block = HunyuanMidBlock(block_out_channels[0], act)
        self.up_blocks = nn.ModuleList([])

        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            if add_spatial_upsample:
                upsample_out_channels = block_out_channels[i + 1]
                self.up_blocks.append(
                    HunyuanUpBlock(input_channel, output_channel, layers_per_block + 1, act, upsample_out_channels)
                )
                input_channel = upsample_out_channels
            else:
                self.up_blocks.append(HunyuanUpBlock(input_channel, output_channel, layers_per_block + 1, act, None))
                input_channel = output_channel

        self.norm_out = HunyuanRMSNorm(block_out_channels[-1])
        self.conv_act = act()
        self.conv_out = HunyuanConv2d(block_out_channels[-1], out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode (b, in_channels, h', w') to (b, out_channels, h, w)."""
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.conv_in(hidden_states) + hidden_states
        hidden_states = self.mid_block(hidden_states)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)
        hidden_states = self.conv_out(self.conv_act(self.norm_out(hidden_states)))
        return hidden_states


class VAEEmbedder(nn.Module):
    """2D VAE-style patch tokenizer for BCAT frames."""

    def __init__(self, config, x_num: int, data_dim: int):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.data_dim = data_dim
        self.x_num = x_num
        self.compression_ratio = config.compression_ratio

        n_down = math.log2(self.compression_ratio)
        assert n_down.is_integer(), f"compression_ratio {self.compression_ratio} must be a power of 2"
        assert x_num % self.compression_ratio == 0, (
            f"x_num must be divisible by compression_ratio, x_num: {x_num}, compression_ratio: {self.compression_ratio}"
        )
        n_down = int(n_down)

        self.patch_num = x_num // self.compression_ratio

        act = get_activation(config.activation)
        block_out_channels = tuple(min(config.hidden_dim * (2**i), config.max_hidden_dim) for i in range(n_down + 1))

        self.encoder = HunyuanEncoder2D(
            in_channels=data_dim,
            out_channels=self.dim,
            block_out_channels=block_out_channels,
            layers_per_block=config.num_res_blocks,
            spatial_compression_ratio=self.compression_ratio,
            act=act,
        )
        self.decoder = HunyuanDecoder2D(
            in_channels=self.dim,
            out_channels=data_dim,
            block_out_channels=tuple(reversed(block_out_channels)),
            layers_per_block=config.num_res_blocks,
            spatial_compression_ratio=self.compression_ratio,
            act=act,
        )

    def encode(self, data: torch.Tensor, skip_len: int = 0) -> torch.Tensor:
        """
        Input:
            data:           Tensor (bs, skip_len+input_len, x_num, x_num, data_dim)
        Output:
            data:           Tensor (bs, input_len, patch_num, patch_num, dim)
        """
        bs = data.size(0)
        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")
        data = self.encoder(data)
        return rearrange(data, "(b t) d h w -> b t h w d", b=bs)

    def decode(self, data_output: torch.Tensor) -> torch.Tensor:
        """
        Input:
            data_output:    Tensor (bs, input_len, patch_num, patch_num, dim)
        Output:
            data_output:    Tensor (bs, input_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data_output = rearrange(data_output, "b t h w d -> (b t) d h w")
        data_output = self.decoder(data_output)
        return rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
