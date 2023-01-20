import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Constructs the encoder sub-network of VAE. This class implements the
    encoder part of Variational Auto-encoder. It will transform primary
    data in the `n_vars` dimension-space to means and log variances of `z_dimension` latent space.

    Parameters
    ----------
    x_dimension: integer
        number of gene expression space dimensions.
    layer_sizes: List
        List of hidden layer sizes.
    z_dimension: integer
        number of latent space dimensions.
    dropout_rate: float
        dropout rate
    """

    def __init__(self, x_dimension: int, layer_sizes: list, z_dimension: int, dropout_rate: float):
        super().__init__() # to run nn.Module's init method

        # encoder architecture
        self.FC = None
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                    self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))

        #self.FC = nn.ModuleList(self.FC)

        print("\tMean/Var Layer in/out:", layer_sizes[-1], z_dimension)
        self.mean_encoder = nn.Linear(layer_sizes[-1], z_dimension)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], z_dimension)


    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)

        mean = self.mean_encoder(x)
        log_var = self.log_var_encoder(x)
        return mean, log_var

class Decoder(nn.Module):
    """
            Constructs the decoder sub-network of VAE. This class implements the
            decoder part of Variational Auto-encoder. Decodes data from latent space to data space. It will transform constructed latent space to the previous space of data with n_dimensions = n_vars.

            # Parameters
               z_dimension: integer
               number of latent space dimensions.
               layer_sizes: List
               List of hidden layer sizes.
               x_dimension: integer
               number of gene expression space dimensions.
               dropout_rate: float
               dropout rate


        """
    def __init__(self, z_dimension: int, layer_sizes: list, x_dimension: int, dropout_rate: float):
        super().__init__()

        layer_sizes = [z_dimension] + layer_sizes
        # decoder architecture
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out", layer_sizes[0], layer_sizes[1])
        self.FirstL.add_module(name="L0", module=nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1]))
        self.FirstL.add_module(name="A0", module=nn.LeakyReLU(negative_slope=0.3))
        self.FirstL.add_module(name="D0", module=nn.Dropout(p=dropout_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.LeakyReLU(negative_slope=0.3))
                    self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dropout_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))


    def forward(self, z: torch.Tensor):
        dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        recon_x = self.recon_decoder(x)
        return recon_x
