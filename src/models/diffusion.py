import torch
import torch.nn as nn
from einops import rearrange

import diffusers
from diffusers.pipelines import DiffusionPipeline
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel

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
            num_train_timesteps=config.scheduler.num_steps,
            beta_schedule=config.scheduler.beta_schedule,
            prediction_type=config.scheduler.prediction_type,
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
        elif mode == "rollout":
            return self.rollout(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, data_label, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            data_label:    Tensor     (bs, output_len=1, x_num, x_num, data_dim)

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

        match self.config.scheduler.prediction_type:
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

        match self.config.scheduler.reverse_type:
            case "ddpm":
                pipeline = DDPMCondPipeline(unet=self.unet2d, scheduler=self.noise_scheduler)
                n_steps = self.config.scheduler.num_steps
            case "ddim":
                pipeline = DDIMCondPipeline(unet=self.unet2d, scheduler=self.noise_scheduler)
                n_steps = self.config.scheduler.ddim_num_steps

        data_output = pipeline(condition=data_input, num_inference_steps=n_steps)

        data_output = rearrange(data_output, "b c x y -> b 1 x y c")
        # data_output = data_output * data_mask
        return data_output

    @torch.compiler.disable()
    def rollout(self, data_input, normalizer, n_total_steps, data_mask, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len=1, x_num, x_num, data_dim)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
        Output:
            data_output:     Tensor   (bs, output_len, x_num, x_num, data_dim)
        """
        bs, _, x_num, _, data_dim = data_input.size()
        data_all = torch.zeros(
            bs, n_total_steps, x_num, x_num, data_dim, device=data_input.device, dtype=data_input.dtype
        )
        data_all[:, 0:1] = data_input

        match self.config.scheduler.reverse_type:
            case "ddpm":
                pipeline = DDPMCondPipeline(unet=self.unet2d, scheduler=self.noise_scheduler)
                n_steps = self.config.scheduler.num_steps
            case "ddim":
                pipeline = DDIMCondPipeline(unet=self.unet2d, scheduler=self.noise_scheduler)
                n_steps = self.config.scheduler.ddim_num_steps

        for i in range(1, n_total_steps):
            input = data_all[:, i - 1 : i]

            input, _, mean, std = normalizer(input)

            input = rearrange(input, "b 1 x y c -> b c x y")
            output = pipeline(condition=input, num_inference_steps=n_steps)
            output = rearrange(output, "b c x y -> b 1 x y c")

            # output = output * data_mask
            output = output * std + mean
            data_all[:, i : i + 1] = output

        return data_all[:, 1:]


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


class DDIMCondPipeline(DiffusionPipeline):
    """
    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        condition,
        generator=None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output=None,
    ):
        """
        condition: Tensor (b, c, x, y)
        eta:    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
        num_inference_steps:
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
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

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

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
