from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .modules import MaskedLinearDecoder
from ..trvae.modules import Encoder
from ..trvae.losses import mse, nb
from ..trvae._utils import one_hot_encoder
from scarches.models.base._base import CVAELatentsModelMixin


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
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse' or 'nb'.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       mask: Tensor or None
            if not None, Tensor of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
            Automatically sets recon_loss to 'mse'.
       use_decoder_relu: Boolean
            Use ReLU after the linear layer in the interpretable (masked) linear decoder.
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
                 decoder_last_layer: str = "softmax",
                 use_decoder_relu: bool = False,
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb", "zinb"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

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

        self.encoder = Encoder(encoder_layer_sizes,
                               self.latent_dim,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)

        self.decoder = MaskedLinearDecoder(self.latent_dim,
                                           self.input_dim,
                                           self.n_conditions,
                                           mask,
                                           self.recon_loss,
                                           self.decoder_last_layer,
                                           use_decoder_relu)

        if self.use_l_encoder:
            self.l_encoder = Encoder([self.input_dim, 128],
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

        return recon_loss, kl_div, torch.tensor(0.)
