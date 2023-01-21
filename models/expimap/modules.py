import torch
import torch.nn as nn
import numpy as np

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
        n_ext: int = 0,
        n_ext_m: int = 0,
        mask: Optional[torch.Tensor] = None,
        ext_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        if mask is None:
            self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        else:
            self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)

        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

        if self.n_ext != 0:
            self.ext_L = nn.Linear(self.n_ext, n_out, bias=False)

        if self.n_ext_m != 0:
            if ext_mask is not None:
                self.ext_L_m = MaskedLinear(self.n_ext_m, n_out, ext_mask, bias=False)
            else:
                self.ext_L_m = nn.Linear(self.n_ext_m, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            expr, cond = x, None
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)

        if self.n_ext == 0:
            ext = None
        else:
            expr, ext = torch.split(expr, [expr.shape[1] - self.n_ext, self.n_ext], dim=1)

        if self.n_ext_m == 0:
            ext_m = None
        else:
            expr, ext_m = torch.split(expr, [expr.shape[1] - self.n_ext_m, self.n_ext_m], dim=1)

        out = self.expr_L(expr)
        if ext is not None:
            out = out + self.ext_L(ext)
        if ext_m is not None:
            out = out + self.ext_L_m(ext_m)
        if cond is not None:
            out = out + self.cond_L(cond)
        return out


class MaskedLinearDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_cond, mask, ext_mask, recon_loss,
                 last_layer=None, n_ext=0, n_ext_m=0):
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
        print("\tMasked linear layer in, ext_m, ext, cond, out: ", in_dim, n_ext_m, n_ext, n_cond, out_dim)
        if mask is not None:
            print('\twith hard mask.')
        else:
            print('\twith soft mask.')

        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        self.n_cond = 0
        if n_cond is not None:
            self.n_cond = n_cond

        self.L0 = MaskedCondLayers(in_dim, out_dim, n_cond, bias=False, n_ext=n_ext, n_ext_m=n_ext_m,
                                   mask=mask, ext_mask=ext_mask)

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

        print("Last Decoder layer:", last_layer)

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
        nz = (v.norm(p=1, dim=0)>0).cpu().numpy()
        nz = np.append(nz, np.full(self.n_ext_m, True))
        nz = np.append(nz, np.full(self.n_ext, True))
        return nz

    def n_inactive_terms(self):
        n = (~self.nonzero_terms()).sum()
        return int(n)


class ExtEncoder(nn.Module):
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: Optional[int] = None,
                 n_expand: int = 0):
        super().__init__()
        self.n_classes = 0
        self.n_expand = n_expand
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out and cond:", in_size, out_size, self.n_classes)
                    self.FC.add_module(name="L{:d}".format(i), module=MaskedCondLayers(in_size,
                                                                                       out_size,
                                                                                       self.n_classes,
                                                                                       bias=True))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
                if use_bn:
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                elif use_ln:
                    self.FC.add_module("N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

        if self.n_expand != 0:
            print("\tExpanded Mean/Var Layer in/out:", layer_sizes[-1], self.n_expand)
            self.expand_mean_encoder = nn.Linear(layer_sizes[-1], self.n_expand)
            self.expand_var_encoder = nn.Linear(layer_sizes[-1], self.n_expand)

    def forward(self, x, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)

        if self.n_expand != 0:
            means = torch.cat((means, self.expand_mean_encoder(x)), dim=-1)
            log_vars = torch.cat((log_vars, self.expand_var_encoder(x)), dim=-1)
        return means, log_vars
