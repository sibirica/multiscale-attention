import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    """

    def __init__(
        self,
        hidden_size=1024,  # 32
        num_blocks=8,
        sparsity_threshold=-1,  # 0.01
        modes=32,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        act=nn.GELU(),
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor

        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)  # 0.02

        self.act = act

        self.w1 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    def forward(self, x):
        B, H, W, C = x.shape
        dtype = x.dtype
        x = x.float()

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o1_imag = torch.zeros(
            [B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[1])
            + self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        if self.sparsity_threshold > 0:
            x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        return x.type(dtype)
