import torch
import torch.nn as nn

from typing import Optional

from ..trvae._utils import one_hot_encoder

class MaskedLinear(nn.Linear):
    def __init__(self, n_in,  n_out, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        if n_in != mask.shape[0] or n_out != mask.shape[1]:
            raise ValueError('Incorrect shape of the mask.')

        super().__init__(n_in, n_out, bias)

        self.register_buffer('mask', mask.t())

        # zero out the weights for group lasso
        # gradient descent won't change these zero weights
        self.weight.data*=self.mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight*self.mask, self.bias)


class MaskedCondLayers(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cond: int,
            bias: bool,
            mask: torch.Tensor
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class MaskedLinearDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, n_cond, mask, recon_loss, last_layer=None,
                 use_relu=False):
        super().__init__()

        if recon_loss == "mse":
            if last_layer == "softmax":
                raise ValueError("Can't specify softmax last layer with mse loss.")
            last_layer = "identity" if last_layer is None else last_layer
        elif recon_loss == "nb":
            last_layer = "softmax" if last_layer is None else last_layer
        else:
            raise ValueError("Unrecognized loss.")

        print("Decoder Architecture:")
        print("\tMasked linear layer in, out and cond: ", in_dim, out_dim, n_cond)

        self.n_cond = 0
        if n_cond is not None:
            self.n_cond = n_cond

        self.L0 = MaskedCondLayers(in_dim, out_dim, n_cond, bias=False, mask=mask)

        if last_layer == "softmax":
            self.mean_decoder = nn.Softmax(dim=-1)
        elif last_layer == "softplus":
            self.mean_decoder = nn.Softplus()
        elif last_layer == "exp":
            self.mean_decoder = torch.exp
        elif last_layer == "relu":
            self.mean_decoder = nn.ReLU()
        elif last_layer == "identity":
            self.mean_decoder = lambda a: a
        else:
            raise ValueError("Unrecognized last layer.")

        print("Last layer:", last_layer)

    def forward(self, z, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_cond)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.L0(z_cat)
        else:
            dec_latent = self.L0(z)

        recon_x = self.mean_decoder(dec_latent)

        return recon_x, dec_latent

    def nonzero_terms(self):
        v = self.L0.expr_L.weight.data
        return (v.norm(p=2, dim=0)>0).cpu().numpy()

    def n_inactive_terms(self):
        n = (~self.nonzero_terms()).sum()
        return int(n)
