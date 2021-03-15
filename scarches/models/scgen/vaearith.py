import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy
from scipy import sparse

from .modules import Encoder, Decoder
from .util import balancer, extractor

class vaeArith(nn.Module):
    """ScArches model class. This class contains the implementation of Variational Auto-encoder network with Vector Arithmetics.

       Parameters
       ----------
       x_dim: Integer
            Number of input features (i.e. number of gene expression space dimensions).
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Size of the bottleneck layer (z).
       dr_rate: Float
            Dropout rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
    """

    def __init__(self, x_dim: int, hidden_layer_sizes: list = [128,128], z_dimension: int = 10, dr_rate: float = 0.05, **kwargs):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(z_dimension, int)


        self.x_dim = x_dim
        self.z_dim = z_dimension
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dr_rate = dr_rate

        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.x_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.x_dim)

        self.encoder = Encoder(self.x_dim, encoder_layer_sizes, self.z_dim, self.dr_rate)
        self.decoder = Decoder(self.z_dim, decoder_layer_sizes, self.x_dim, self.dr_rate)


        self.alpha = kwargs.get("alpha", 0.0000001)


    @staticmethod
    def _sample_z(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Samples from standard Normal distribution with shape [size, z_dim] and
        applies re-parametrization trick. It is actually sampling from latent
        space distributions with N(mu, var) computed by the Encoder.

        Parameters
        ----------
        mean:
            Mean of the latent Gaussian
        log_var:
            Standard deviation of the latent Gaussian
        Returns
        -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        The computed Tensor of samples with shape [size, z_dim].
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def to_latent(self, data: torch.Tensor) -> torch.Tensor:
        """
        Map `data` in to the latent space. This function will feed data
        in encoder part of VAE and compute the latent space coordinates
        for each sample in data.

        Parameters
        ----------
        data:
            Torch Tensor to be mapped to latent space. `data.X` has to be in shape [n_obs, x_dim].

        Returns
        -------
        latent:
            Returns Torch Tensor containing latent space encoding of 'data'
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data) # to tensor
        mu, logvar = self.encoder(data)
        latent = self._sample_z(mu, logvar)
        return latent

    def _avg_vector(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the average of points which computed from mapping `data`
        to encoder part of VAE.

        Parameters
        ----------
        data:
            Torch Tensor matrix to be mapped to latent space. Note that `data.X` has to be in shape [n_obs, x_dim].

        Returns
        -------
            The average of latent space mapping in Torch Tensor

        """
        latent = self.to_latent(data)
        latent_avg = torch.mean(latent, dim=0) # maybe keepdim = True, so that shape (,1)
        return latent_avg

    def reconstruct(self, data, use_data=False) -> torch.Tensor:
        """
        Map back the latent space encoding via the decoder.

        Parameters
        ----------
        data: `~anndata.AnnData`
            Annotated data matrix whether in latent space or gene expression space.
        use_data: bool
            This flag determines whether the `data` is already in latent space or not.
            if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
            if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, x_dim]).

        Returns
        -------
        rec_data:
            Returns Torch Tensor containing reconstructed 'data' in shape [n_obs, x_dim].
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data) # to tensor

        if use_data:
            latent = data
        else:
            latent = self.to_latent(data)

        rec_data = self.decoder(latent)
        return rec_data

    def _loss_function(self, x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Defines the loss function of VAE network after constructing the whole
        network. This will define the KL Divergence and Reconstruction loss for
        VAE. The VAE Loss will be weighted sum of reconstruction loss and KL Divergence loss.

        Parameters
        ----------

        Returns
        -------
        Returns VAE Loss as Torch Tensor.
        """
        kl_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) # check dimensions
        recons_loss = F.mse_loss(xhat, x)
        vae_loss = recons_loss + self.alpha * kl_loss
        return vae_loss


    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self._sample_z(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None, obs_key="all"):
        """
        Predicts the cell type provided by the user in stimulated condition.

        Parameters
        ----------
        celltype_to_predict: basestring
            The cell type you want to be predicted.
        obs_key: basestring or dict
            Dictionary of celltypes you want to be observed for prediction.
        adata_to_predict: `~anndata.AnnData`
            Adata for unpertubed cells you want to be predicted.

        Returns
        -------
        predicted_cells: Torch Tensor
            `Torch Tensor` of predicted cells in primary space.
        delta: Torch Tensor
            Difference between stimulated and control cells in latent space
        """
        device = next(self.parameters()).device # get device of model.parameters
        if obs_key == "all":
            ctrl_x = adata[adata.obs[condition_key] == conditions["ctrl"], :]
            stim_x = adata[adata.obs[condition_key] == conditions["stim"], :]
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        else:
            key = list(obs_key.keys())[0]
            values = obs_key[key]
            subset = adata[adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == conditions["ctrl"], :]
            stim_x = subset[subset.obs[condition_key] == conditions["stim"], :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict

        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
            latent_ctrl = self._avg_vector(torch.tensor(ctrl_x.X.A[cd_ind, :], device=device))
            latent_sim = self._avg_vector(torch.tensor(stim_x.X.A[stim_ind, :], device=device))
        else:
            latent_ctrl = self._avg_vector(torch.tensor(ctrl_x.X[cd_ind, :], device=device))
            latent_sim = self._avg_vector(torch.tensor(stim_x.X[stim_ind, :], device=device))

        delta = latent_sim - latent_ctrl
        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent(torch.tensor(ctrl_pred.X.A, device=device))
        else:
            latent_cd = self.to_latent(torch.tensor(ctrl_pred.X, device=device))

        stim_pred = delta + latent_cd
        predicted_cells = self.reconstruct(stim_pred, use_data=True)
        return predicted_cells, delta
