from logging import getLogger
import torch
from tabulate import tabulate
from ema_pytorch import EMA

from .bcat import BCAT, BCAT_Reg
from .causal import BCAT_causal
from .baselines import FNO, UNet, ViT, DeepONet
from .space_time import ST_auto
from .vq_bcat import VQBCAT
from .vq.vqvae import VQModelWrapper
from .diffusion import I2IDiffusion

logger = getLogger()


def build_model(params, model_config, data_config, symbol_env):

    modules = {}

    # get model
    name = model_config.name

    if name == "st_auto":
        modules["model"] = ST_auto(model_config, data_config.x_num, data_config.max_output_dimension, data_config.t_num)

    elif name == "vq_bcat_auto":
        modules["model"] = VQBCAT(model_config, data_config.x_num, data_config.max_output_dimension, data_config.t_num)

    elif name == "vqvae":
        config = model_config.embedder
        modules["model"] = VQModelWrapper(
            n_embed=config.codebook_size,
            embed_dim=config.dim,
            z_ch=config.z_ch,
            in_ch=data_config.max_output_dimension,
            mid_ch=config.mid_ch,
            ch_mult=config.ch_mult,
        )

    elif name == "bcat_reg_auto":
        modules["model"] = BCAT_Reg(
            model_config, data_config.x_num, data_config.max_output_dimension, data_config.t_num
        )

    elif name == "bcat_auto":
        modules["model"] = BCAT(model_config, data_config.x_num, data_config.max_output_dimension, data_config.t_num)

    elif name == "bcat_next_token_auto":
        modules["model"] = BCAT_causal(
            model_config, data_config.x_num, data_config.max_output_dimension, data_config.t_num
        )

    elif name == "diffusion2d":
        modules["model"] = I2IDiffusion(model_config, data_config.x_num, data_config.max_output_dimension)

    elif name == "fno":
        modules["model"] = FNO(model_config, data_config.max_output_dimension)

    elif name == "vit":
        modules["model"] = ViT(model_config, data_config.x_num, data_config.max_output_dimension)

    elif name == "unet":
        modules["model"] = UNet(model_config, data_config.max_output_dimension)

    elif name == "deeponet":
        modules["model"] = DeepONet(model_config, data_config, params.input_len)

    else:
        assert False, f"Model {name} hasn't been implemented"

    if params.ema.enable and (params.is_master or (not params.eval_only)):
        logger.info(f"Using EMA for model parameters")
        modules["model_ema"] = EMA(
            modules["model"],
            beta=params.ema.beta,
            update_after_step=params.ema.update_after_step,
            update_every=params.ema.update_every,
            power=params.ema.power,
            include_online_model=False,
        )
    else:
        params.ema.enable = False

    # reload pretrained modules
    if params.reload_model:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model, weights_only=False)
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"

            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()}
            if all([k2.startswith("_orig_mod.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len("_orig_mod.") :]: v2 for k2, v2 in reloaded[k].items()}
            v.load_state_dict(reloaded[k])

    if params.ema.enable and params.eval_only:
        modules["model"].load_state_dict(modules["model_ema"].ema_model.state_dict())
        del modules["model_ema"]

    # log
    for k, v in modules.items():
        if k.endswith("_ema"):
            continue
        logger.info(f"{k}: {v}")

    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad]):,}"
        if hasattr(v, "summary"):
            # for individual components of a wrapper model
            s += v.summary()
        logger.info(s)

    # for k, v in modules.items():
    #     table_data = [(name, str(param.shape), param.requires_grad) for name, param in v.named_parameters()]
    #     logger.info("\n" + tabulate(table_data, headers=["Parameter Name", "Shape", "Requires Grad"], tablefmt="grid"))
    #     table_data = [(name, str(param.shape)) for name, param in v.named_parameters() if param.requires_grad]
    #     logger.info("\n" + tabulate(table_data, headers=["Trainable Parameters", "Shape"], tablefmt="grid"))

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    if params.compile:
        for k, v in modules.items():
            if k.endswith("_ema"):
                continue

            # modules[k] = torch.compile(v, mode="reduce-overhead")
            modules[k] = torch.compile(v)

    return modules
