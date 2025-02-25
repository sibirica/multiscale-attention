import torch
import torch.nn as nn
from einops import rearrange

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from logging import getLogger

logger = getLogger()


class I2IDiffusion(nn.Module):
    def __init__(self, config, x_num, max_output_dim):
        super().__init__()

        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.unet2d = UNet2DModel(
            sample_size=x_num,
            in_channels=max_output_dim * 2,
            out_channels=max_output_dim,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.ddpm_num_steps,
            beta_schedule=config.ddpm_beta_schedule,
            prediction_type=config.prediction_type,
        )

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

    def fwd(self, data_input, data_label, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            data_label:    Tensor     (bs, ouputput_len=1, x_num, x_num, data_dim)

        Output:

        """
        bs = data_input.shape[0]

        condition = rearrange(data_input, "b 1 x y c -> b c x y")
        clean_images = rearrange(data_label, "b 1 x y c -> b c x y")

        noise = torch.randn_like(clean_images)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # forward diffusion process
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # add condition (just simple concat)
        model_input = torch.cat([noisy_images, condition], dim=1)

        model_output = self.unet2d(model_input, timesteps).sample

        return model_output, noise

    @torch.compiler.disable()
    def generate(self, data_input, data_mask, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
        Output:
            data_output:     Tensor   (bs, output_len=1, x_num, x_num, data_dim)
        """
        raise NotImplementedError
