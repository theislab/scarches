import torch
import torch.nn as nn

from ._utils import one_hot_encoder


class CondLayers(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cond: int,
            bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, x.shape[1] - self.n_cond, dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE. It will transform primary space
       input to means and log. variances of latent space with n_dimensions = z_dimension.

       Parameters
       ----------
       layer_sizes: List
            List of first and hidden layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out and cond:", in_size, out_size, self.n_classes)
                    self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size,
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

    def forward(self, x, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)
        return means, log_vars


class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks. It will transform the
       constructed latent space to the previous space of data with n_dimensions = x_dimension.

       Parameters
       ----------
       layer_sizes: List
            List of hidden and last layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 recon_loss: str,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out and cond: ", layer_sizes[0], layer_sizes[1], self.n_classes)
        self.FirstL.add_module(name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False))
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.FirstL.add_module("N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False))
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    elif use_ln:
                        self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.LayerNorm(out_size, elementwise_affine=False))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU())
        if self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))
            # dropout
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent
