import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import VectorQuantizer2 as VectorQuantizer
from .vae import Encoder, Decoder
from einops import rearrange


class VQModel(nn.Module):
    """
    Source: https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py
    """

    def __init__(
        self,
        n_embed,
        embed_dim,
        z_ch,
        remap=None,
        sane_index_shape=True,  # tell vector quantizer to return indices as bhw
        in_ch=3,
        mid_ch=128,
        ch_mult=(1, 2, 2, 4),
        dropout=0,
        using_sa=False,
        using_mid_sa=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed

        ddconfig = dict(
            dropout=dropout,
            in_channels=in_ch,
            ch=mid_ch,
            z_channels=z_ch,
            ch_mult=ch_mult,
            num_res_blocks=2,
            using_sa=using_sa,
            using_mid_sa=using_mid_sa,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(z_ch, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_ch, 1)

    def encode(self, x):
        # (b c h w) -> (b d ph pw)
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        # (b d ph pw) -> (b c h w)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_indices(self, ind):
        # ind: (b ph pw)
        quant = self.quantize.embedding(ind).permute(0, 3, 1, 2)  # (b d ph pw)
        dec = self.decode(quant)
        return dec  # (b c h w)

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VQModelWrapper(VQModel):
    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        input = kwargs["input"]  # (b t x y c)
        bs = input.size(0)
        input = rearrange(input, "b t x y c -> (b t) c x y")

        if mode == "fwd":
            dec, diff = self.fwd(input)
            output = rearrange(dec, "(b t) c x y -> b t x y c", b=bs)
            return output, diff
        elif mode == "generate":
            dec, diff = self.fwd(input)
            output = rearrange(dec, "(b t) c x y -> b t x y c", b=bs)
            return output
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, input):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
