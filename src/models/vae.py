import math

import torch
import torch.nn as nn
from einops import rearrange

from .attention_utils import get_activation


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"Could not find a valid GroupNorm group count for {channels} channels")


class ResBlock(nn.Module):
    """Residual 2D convolution block used by the VAE tokenizer."""

    def __init__(self, channels: int, act: type[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            act(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            act(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply a residual convolution block."""
        return data + self.net(data)


class AttentionBlock(nn.Module):
    """Spatial self-attention block for low-resolution VAE feature maps."""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply full spatial self-attention over HxW tokens."""
        bs, channels, height, width = data.shape
        qkv = self.qkv(self.norm(data))
        qkv = qkv.reshape(bs, 3, self.num_heads, self.head_dim, height * width)
        query, key, value = qkv.unbind(dim=1)
        attn = torch.einsum("bhdn,bhdm->bhnm", query, key) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        data_attn = torch.einsum("bhnm,bhdm->bhdn", attn, value)
        data_attn = data_attn.reshape(bs, channels, height, width)
        return data + self.proj(data_attn)


class DownsampleBlock(nn.Module):
    """Residual block followed by 2x spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        act: type[nn.Module],
        use_attention: bool,
        num_heads: int,
    ):
        super().__init__()
        blocks = [ResBlock(in_channels, act) for _ in range(num_res_blocks)]
        if use_attention:
            blocks.append(AttentionBlock(in_channels, num_heads))
        blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Downsample a spatial feature map by 2."""
        return self.net(data)


class UpsampleBlock(nn.Module):
    """Residual block followed by 2x spatial upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        act: type[nn.Module],
        use_attention: bool,
        num_heads: int,
    ):
        super().__init__()
        blocks = [ResBlock(in_channels, act) for _ in range(num_res_blocks)]
        blocks.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if use_attention:
            blocks.append(AttentionBlock(out_channels, num_heads))
        self.net = nn.Sequential(*blocks)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Upsample a spatial feature map by 2."""
        return self.net(data)


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

        self.patch_num = x_num // self.compression_ratio
        self.patch_num_output = self.patch_num
        self.patch_resolution = self.compression_ratio
        self.patch_resolution_output = self.compression_ratio

        act = get_activation(config.activation)
        n_down = int(n_down)
        attention_resolutions = set(config.attention_resolutions)
        channels = [min(config.hidden_dim * (2**i), config.max_hidden_dim) for i in range(n_down + 1)]

        encoder = [nn.Conv2d(data_dim, channels[0], kernel_size=3, padding=1)]
        resolution = x_num
        for i in range(n_down):
            encoder.append(
                DownsampleBlock(
                    channels[i],
                    channels[i + 1],
                    config.num_res_blocks,
                    act,
                    resolution in attention_resolutions,
                    config.attn_heads,
                )
            )
            resolution //= 2
        encoder.extend(
            [
                ResBlock(channels[-1], act),
                AttentionBlock(channels[-1], config.attn_heads),
                ResBlock(channels[-1], act),
                nn.GroupNorm(_group_count(channels[-1]), channels[-1]),
                act(),
                nn.Conv2d(channels[-1], self.dim, kernel_size=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder)

        decoder = [
            nn.Conv2d(self.dim, channels[-1], kernel_size=1),
            ResBlock(channels[-1], act),
            AttentionBlock(channels[-1], config.attn_heads),
            ResBlock(channels[-1], act),
        ]
        resolution = self.patch_num
        for i in range(n_down, 0, -1):
            resolution *= 2
            decoder.append(
                UpsampleBlock(
                    channels[i],
                    channels[i - 1],
                    config.num_res_blocks,
                    act,
                    resolution in attention_resolutions,
                    config.attn_heads,
                )
            )
        decoder.extend(
            [
                nn.GroupNorm(_group_count(channels[0]), channels[0]),
                act(),
                nn.Conv2d(channels[0], data_dim, kernel_size=3, padding=1),
            ]
        )
        self.decoder = nn.Sequential(*decoder)

    def encode(self, data: torch.Tensor, skip_len: int = 0) -> torch.Tensor:
        """
        Input:
            data:           Tensor (bs, skip_len+input_len, x_num, x_num, data_dim)
        Output:
            data:           Tensor (bs, input_len, x_num, x_num, dim)
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
