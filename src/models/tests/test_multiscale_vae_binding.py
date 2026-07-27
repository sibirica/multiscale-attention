import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import torch.nn.attention as _torch_attn


@contextmanager
def _no_op_sdpa_kernel(*args, **kwargs):
    yield


_torch_attn.sdpa_kernel = _no_op_sdpa_kernel
import models.bcat as _bcat_mod
import models.multiscale_bcat as _ms_mod

_bcat_mod.sdpa_kernel = _no_op_sdpa_kernel
_ms_mod.sdpa_kernel = _no_op_sdpa_kernel

from models.multiscale_bcat import MultiscaleBCAT
from models.bcat import BCAT

T_NUM = 24
INPUT_LEN = 1
X_NUM = 16
DATA_DIM = 3
DIM_EMB = 32
DIM_FFN = 64
N_HEAD = 4
N_LAYER = 6
RATE = 10
COMPRESSION = 4  # x_num//compression = patch_num = 4 -> seq_len_per_step 16


def vae_embedder_cfg():
    return dict(
        type="vae",
        dim=DIM_EMB,
        time_embed="learnable",
        max_time_len=T_NUM + 4,
        compression_ratio=COMPRESSION,
        hidden_dim=16,
        max_hidden_dim=64,
        num_res_blocks=1,
        activation="silu",
    )


def conv_embedder_cfg():
    return dict(
        type="conv",
        dim=DIM_EMB,
        patch_num=X_NUM // COMPRESSION,
        patch_num_output=X_NUM // COMPRESSION,
        time_embed="learnable",
        max_time_len=T_NUM + 4,
        conv_dim=8,
        early_conv=False,
        deep=False,
    )


def ms_config(embedder):
    return OmegaConf.create(
        dict(
            name="multiscale_bcat_auto",
            n_layer=N_LAYER,
            dim_emb=DIM_EMB,
            slow_dim=DIM_EMB,
            dim_ffn=DIM_FFN,
            pool_dim=DIM_FFN,
            dropout=0.0,
            attn_dropout=0.0,
            n_head=N_HEAD,
            norm_first=True,
            qk_norm=True,
            norm="layer",
            activation="swiglu",
            recombine_activation="swiglu",
            ln_mode="keel",
            keel_alpha=12,
            rotary=False,
            flex_attn=False,
            kv_cache=False,
            act_ckpt=False,
            logit_softcap=0,
            rate=RATE,
            shared_scale_ffn=False,
            limit_window=True,
            attn_sink_tokens=1,
            self_window=4,
            fast_to_slow_window=2,
            slow_to_fast_window=RATE,
            embedder=embedder,
        )
    )


def bcat_config(embedder):
    return OmegaConf.create(
        dict(
            name="bcat_auto",
            n_layer=N_LAYER,
            dim_emb=DIM_EMB,
            dim_ffn=DIM_FFN,
            dropout=0.0,
            attn_dropout=0.0,
            n_head=N_HEAD,
            norm_first=True,
            qk_norm=False,
            norm="layer",
            ln_mode="pre",
            activation="gelu",
            rotary=False,
            bias=True,
            flex_attn=False,
            kv_cache=False,
            act_ckpt=False,
            logit_softcap=0,
            embedder=embedder,
        )
    )


def run(label, model, steps=200, lr=1e-3):
    torch.manual_seed(1)
    data = torch.randn(2, T_NUM, X_NUM, X_NUM, DATA_DIM)
    times = torch.linspace(0, 1, T_NUM).view(1, T_NUM, 1).expand(2, -1, -1).contiguous()
    label_data = data[:, INPUT_LEN:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for i in range(steps):
        opt.zero_grad()
        out = model("fwd", data=data, times=times, input_len=INPUT_LEN)
        loss = F.mse_loss(out, label_data)
        loss.backward()
        opt.step()
        if i % (steps // 10) == 0 or i == steps - 1:
            losses.append((i, loss.item()))
    print(f"\n== {label} ==")
    for i, l in losses:
        print(f"  step {i:4d}  loss {l:.5f}")
    # gradient flow report on a fresh backward
    opt.zero_grad()
    out = model("fwd", data=data, times=times, input_len=INPUT_LEN)
    F.mse_loss(out, label_data).backward()
    grad_none = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"  params with grad=None: {len(grad_none)}")
    for n in grad_none[:20]:
        print(f"    {n}")
    return losses[-1][1]


GEN_INPUT_LEN = 10


@torch.no_grad()
def fwd_vs_generate(label, cfg, kv=False, is_bcat=False, amp=False):
    torch.manual_seed(3)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.kv_cache = kv
    if is_bcat:
        model = BCAT(cfg, X_NUM, DATA_DIM, max_data_len=T_NUM)
    else:
        model = MultiscaleBCAT(cfg, X_NUM, DATA_DIM, max_data_len=T_NUM)
    model.eval()
    dtype = torch.bfloat16 if amp else torch.float32
    if kv:
        model.setup_cache(2, dtype)
    data = torch.randn(2, T_NUM, X_NUM, X_NUM, DATA_DIM)
    times = torch.linspace(0, 1, T_NUM).view(1, T_NUM, 1).expand(2, -1, -1).contiguous()
    data_mask = torch.ones(1, 1, 1, 1, DATA_DIM)

    with torch.amp.autocast("cpu", enabled=amp, dtype=torch.bfloat16):
        out_fwd = model("fwd", data=data, times=times, input_len=GEN_INPUT_LEN)  # (bs, output_len, ...)
        out_gen = model(
            "generate",
            data_input=data[:, :GEN_INPUT_LEN],
            times=times,
            input_len=GEN_INPUT_LEN,
            data_mask=data_mask,
        )
    diff0 = (out_fwd[:, 0].float() - out_gen[:, 0].float()).abs().max().item()
    scale = out_fwd[:, 0].float().abs().mean().item()
    print(f"\n== fwd-vs-generate: {label} (amp={amp}) ==")
    print(f"  out_fwd len={out_fwd.size(1)} out_gen len={out_gen.size(1)} mean|fwd|={scale:.3e}")
    print(f"  max|fwd[0]-gen[0]| = {diff0:.3e}")


def run_muon(label, model, steps=200, lr=1e-3):
    from utils.muon import Muon

    torch.manual_seed(1)
    data = torch.randn(2, T_NUM, X_NUM, X_NUM, DATA_DIM)
    times = torch.linspace(0, 1, T_NUM).view(1, T_NUM, 1).expand(2, -1, -1).contiguous()
    label_data = data[:, INPUT_LEN:]

    named_params = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    adam_keys = ["embedding"]
    muon_params, adam_params = [], []
    for n, p in named_params:
        if p.ndim < 2 or any(s in n for s in adam_keys):
            adam_params.append(p)
        else:
            muon_params.append(p)
    opt = Muon(lr=lr, wd=0.0, muon_params=muon_params, adamw_params=adam_params)
    model.train()
    losses = []
    for i in range(steps):
        opt.zero_grad()
        out = model("fwd", data=data, times=times, input_len=INPUT_LEN)
        loss = F.mse_loss(out, label_data)
        loss.backward()
        opt.step()
        if i % (steps // 10) == 0 or i == steps - 1:
            losses.append((i, loss.item()))
    print(f"\n== [MUON] {label} ==")
    for i, l in losses:
        print(f"  step {i:4d}  loss {l:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)

    if os.environ.get("RUN_MUON"):
        run_muon(
            "MultiscaleBCAT + vae", MultiscaleBCAT(ms_config(vae_embedder_cfg()), X_NUM, DATA_DIM, max_data_len=T_NUM)
        )
        run_muon(
            "MultiscaleBCAT + conv", MultiscaleBCAT(ms_config(conv_embedder_cfg()), X_NUM, DATA_DIM, max_data_len=T_NUM)
        )

    # fwd (teacher forcing) vs generate (autoregressive) consistency.
    # For a correct model these must match on shared-context predictions.
    fwd_vs_generate("MultiscaleBCAT + vae (kv=0)", ms_config(vae_embedder_cfg()), kv=False)
    fwd_vs_generate("MultiscaleBCAT + conv (kv=0)", ms_config(conv_embedder_cfg()), kv=False)
    fwd_vs_generate("MultiscaleBCAT + vae (kv=1)", ms_config(vae_embedder_cfg()), kv=True)
    fwd_vs_generate("BCAT + vae", bcat_config(vae_embedder_cfg()), kv=False, is_bcat=True)

    # bf16 autocast (matches production amp=1)
    fwd_vs_generate("MultiscaleBCAT + vae (kv=1)", ms_config(vae_embedder_cfg()), kv=True, amp=True)
    fwd_vs_generate("MultiscaleBCAT + conv (kv=1)", ms_config(conv_embedder_cfg()), kv=True, amp=True)
