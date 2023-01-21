from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .modules import MaskedLinearDecoder, ExtEncoder
from ..trvae.losses import mse, nb
from .losses import hsic
from ..trvae._utils import one_hot_encoder
from ..base._base import CVAELatentsModelMixin


class expiMap(nn.Module, CVAELatentsModelMixin):
    """ScArches model class. This class contains the implementation of Conditional Variational Auto-encoder.
       Parameters
       ----------
       input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z) size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`=0 no dropout will be applied.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse' or 'nb'.
       use_l_encoder: Boolean
            If True and `decoder_last_layer`='softmax', libary size encoder is used.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       mask: Tensor or None
            if not None, Tensor of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
       decoder_last_layer: String or None
            The last layer of the decoder. Must be 'softmax' (default for 'nb' loss), identity(default for 'mse' loss),
            'softplus', 'exp' or 'relu'.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 mask: torch.Tensor,
                 conditions: list,
                 hidden_layer_sizes: list = [256, 256],
                 dr_rate: float = 0.05,
                 recon_loss: str = 'nb',
                 use_l_encoder: bool = False,
                 use_bn: bool = False,
                 use_ln: bool = True,
                 decoder_last_layer: Optional[str] = None,
                 soft_mask: bool = False,
                 n_ext: int = 0,
                 n_ext_m: int = 0,
                 use_hsic: bool = False,
                 hsic_one_vs_all: bool = False,
                 ext_mask: Optional[torch.Tensor] = None,
                 soft_ext_mask: bool = False
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb"], "'recon_loss' must be 'mse' or 'nb'"

        print("\nINITIALIZING NEW NETWORK..............")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.recon_loss = recon_loss
        self.freeze = False
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.use_mmd = False

        self.n_ext_encoder = n_ext + n_ext_m
        self.n_ext_decoder = n_ext
        self.n_ext_m_decoder = n_ext_m

        self.use_hsic = use_hsic and self.n_ext_decoder > 0
        self.hsic_one_vs_all = hsic_one_vs_all

        self.soft_mask = soft_mask and mask is not None
        self.soft_ext_mask = soft_ext_mask and ext_mask is not None

        if decoder_last_layer is None:
            if recon_loss == 'nb':
                self.decoder_last_layer = 'softmax'
            else:
                self.decoder_last_layer = 'identity'
        else:
            self.decoder_last_layer = decoder_last_layer

        self.use_l_encoder = use_l_encoder

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss == "nb":
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)

        self.cell_type_encoder = None

        self.encoder = ExtEncoder(encoder_layer_sizes,
                                  self.latent_dim,
                                  self.use_bn,
                                  self.use_ln,
                                  self.use_dr,
                                  self.dr_rate,
                                  self.n_conditions,
                                  self.n_ext_encoder)

        if self.soft_mask:
            self.n_inact_genes = (1-mask).sum().item()
            soft_shape = mask.shape
            if soft_shape[0] != latent_dim or soft_shape[1] != input_dim:
                raise ValueError('Incorrect shape of the soft mask.')
            self.mask = mask.t()
            mask = None
        else:
            self.mask = None

        if self.soft_ext_mask:
            self.n_inact_ext_genes = (1-ext_mask).sum().item()
            ext_shape = ext_mask.shape
            if ext_shape[0] != self.n_ext_m_decoder:
                raise ValueError('Dim 0 of ext_mask should be the same as n_ext_m_decoder.')
            if ext_shape[1] != self.input_dim:
                raise ValueError('Dim 1 of ext_mask should be the same as input_dim.')
            self.ext_mask = ext_mask.t()
            ext_mask = None
        else:
            self.ext_mask = None

        self.decoder = MaskedLinearDecoder(self.latent_dim,
                                           self.input_dim,
                                           self.n_conditions,
                                           mask,
                                           ext_mask,
                                           self.recon_loss,
                                           self.decoder_last_layer,
                                           self.n_ext_decoder,
                                           self.n_ext_m_decoder)

        if self.use_l_encoder:
            self.l_encoder = ExtEncoder([self.input_dim, 128],
                                        1,
                                        self.use_bn,
                                        self.use_ln,
                                        self.use_dr,
                                        self.dr_rate,
                                        self.n_conditions)

    def forward(self, x=None, batch=None, sizefactor=None, labeled=None):
        x_log = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_log = x

        z1_mean, z1_log_var = self.encoder(x_log, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "nb":
            if self.use_l_encoder and self.decoder_last_layer == "softmax":
                sizefactor = torch.exp(self.sampling(*self.l_encoder(x_log, batch))).flatten()
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            if self.decoder_last_layer == "softmax":
                dec_mean = dec_mean_gamma * size_factor_view
            else:
                dec_mean = dec_mean_gamma
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = kl_divergence(
            Normal(z1_mean, torch.sqrt(z1_var)),
            Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var))
        ).sum(dim=1).mean()

        if self.use_hsic:
            if not self.hsic_one_vs_all:
                z_ann = z1[:, :-self.n_ext_decoder]
                z_ext = z1[:, -self.n_ext_decoder:]
                hsic_loss = hsic(z_ann, z_ext)
            else:
                hsic_loss = 0.
                sz = self.latent_dim + self.n_ext_encoder
                shift = self.latent_dim + self.n_ext_m_decoder
                for i in range(self.n_ext_decoder):
                    sel_cols = torch.full((sz,), True, device=z1.device)
                    sel_cols[shift + i] = False
                    rest = z1[:, sel_cols]
                    term = z1[:, ~sel_cols]
                    hsic_loss = hsic_loss + hsic(term, rest)
        else:
            hsic_loss = torch.tensor(0.0, device=z1.device)

        return recon_loss, kl_div, hsic_loss
