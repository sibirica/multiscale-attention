import torch
import torch.nn as nn
from einops import rearrange

import diffusers
from diffusers.pipelines import DiffusionPipeline
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

        match self.config.prediction_type:
            case "epsilon":
                output_d = {"output": model_output, "noise": noise}
            case "sample":
                alpha_t = _extract_into_tensor(self.noise_scheduler.alphas_cumprod, timesteps, (bs, 1, 1, 1))
                snr_weights = alpha_t / (1 - alpha_t)
                output_d = {"output": model_output, "label": clean_images, "snr_weights": snr_weights}

        return output_d

    @torch.compiler.disable()
    def generate(self, data_input, data_mask, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
        Output:
            data_output:     Tensor   (bs, output_len=1, x_num, x_num, data_dim)
        """
        data_input = rearrange(data_input, "b 1 x y c -> b c x y")
        pipeline = DDPMCondPipeline(unet=self.unet2d, scheduler=self.noise_scheduler)
        data_output = pipeline(condition=data_input, num_inference_steps=self.config.ddpm_num_steps)

        data_output = rearrange(data_output, "b c x y -> b 1 x y c")
        # data_output = data_output * data_mask
        return data_output


class DDPMCondPipeline(DiffusionPipeline):
    """
    Pipeline for conditioned image generation. Modified from diffusers.DDPMPipeline
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        condition,
        generator=None,
        num_inference_steps: int = 1000,
    ):
        """
        condition: Tensor (b, c, x, y)
        """
        # Sample gaussian noise to begin loop
        image = torch.randn_like(condition)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # for t in self.progress_bar(self.scheduler.timesteps):
        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            model_input = torch.cat([image, condition], dim=1)
            model_output = self.unet(model_input, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        return image


# helper function
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
