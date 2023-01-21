import torch
import numpy as np

from anndata import AnnData
from typing import Optional, Union

from .vaearith import vaeArith
from ...trainers import vaeArithTrainer
from ..base._utils import _validate_var_names
from ..base._base import BaseMixin


class scgen(BaseMixin):
    """Model for scArches class. This class contains the implementation of Variational Auto-encoder network with Vector Arithmetics.

       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Size of the bottleneck layer (z).
       dr_rate: Float
            Dropout rate applied to all layers, if `dr_rate` == 0 no dropout will be applied.
    """

    def __init__(self, adata: AnnData, hidden_layer_sizes: list = [128, 128], z_dimension: int = 10, dr_rate: float = 0.05):
        self.adata = adata

        self.x_dim_ = adata.n_vars

        self.z_dim_ = z_dimension
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.dr_rate_ = dr_rate

        self.model = vaeArith(self.x_dim_, self.hidden_layer_sizes_, self.z_dim_, self.dr_rate_)

        self.is_trained_ = False
        self.trainer = None

    def train(self, n_epochs: int = 100, lr: float = 0.001, eps: float = 1e-8, batch_size = 32, **kwargs):
        self.trainer = vaeArithTrainer(self.model, self.adata, batch_size, **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True


    def get_latent(self, data: Optional[np.ndarray] = None):
        """
        Map `data` in to the latent space. This function will feed data
        in encoder part of VAE and compute the latent space coordinates
        for each sample in data.

        Parameters
        ----------
        data:  numpy nd-array
            Numpy nd-array to be mapped to latent space. `data.X` has to be in shape [n_obs, x_dim].

        Returns
        -------
        latent: numpy nd-array
            Returns numpy array containing latent space encoding of 'data'
        """
        device = next(self.model.parameters()).device #get device of model.parameters
        if data is None:
            data = self.adata.X

        data = torch.tensor(data, device=device) # to tensor
        latent = self.model.get_latent(data)
        latent = latent.cpu().detach() # to cpu then detach from the comput.graph
        return np.array(latent)


    def reconstruct(self, data, use_data):
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
        rec_data: 'numpy nd-array'
            Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, x_dim].
        """
        device = next(self.model.parameters()).device
        data = torch.tensor(data, device=device) # to tensor
        rec_data = self.model.reconstruct(data, use_data)
        rec_data = rec_data.cpu().detach()
        return np.array(rec_data)


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
        predicted_cells: numpy nd-array
            `numpy nd-array` of predicted cells in primary space.
        delta: float
            Difference between stimulated and control cells in latent space
        """
        #device = next(self.model.parameters()).device # get device of model.parameters
        #adata_tensor = torch.tensor(adata, device=device) # to tensor
        output = self.model.predict(adata, conditions, cell_type_key, condition_key, adata_to_predict, celltype_to_predict, obs_key)
        prediction = output[0].cpu().detach()
        delta = output[1].cpu().detach()
        return np.array(prediction), np.array(delta)


    def batch_removal(self, adata, batch_key, cell_label_key, return_latent = True):
        """
        Removes batch effect of adata

        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
        batch_key: `str`
            batch label key in  adata.obs
        cell_label_key: `str`
            cell type label key in adata.obs
        return_latent: `bool`
            if `True` returns corrected latent representation

        Returns
        -------
        corrected: `~anndata.AnnData`
            adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent_corrected"].
        """
        corrected = self.model.batch_removal(adata, batch_key, cell_label_key, return_latent)
        return corrected


    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct['x_dim_']:
            raise ValueError("Incorrect var dimension")

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'z_dimension': dct['z_dim_'],
            'dr_rate': dct['dr_rate_'],
        }

        return init_params


    @classmethod
    def map_query_data(cls, corrected_reference: AnnData, query: AnnData, reference_model: Union[str, 'scgen'], batch_key: str = 'study', return_latent = True):
        """
        Removes the batch effect between reference and query data.
        Additional training on query data is not needed.

        Parameters
        ----------
        corrected_reference: `~anndata.AnnData`
           Already corrected reference anndata object
        query: `~anndata.AnnData`
            Query anndata object
        batch_key: `str`
            batch label key in query.obs
        return_latent: `bool`
            if `True` returns corrected latent representation

        Returns
        -------
        integrated: `~anndata.AnnData`
        Returns an integrated query.
        """
        query_batches_labels = query.obs[batch_key].unique().tolist()
        query_adata_by_batches = [query[query.obs[batch_key].isin([batch])].copy() for batch in query_batches_labels]

        reference_query_adata = AnnData.concatenate(*[corrected_reference, query_adata_by_batches],
                                                    batch_key="reference_map",
                                                    batch_categories= ['reference'] + query_batches_labels,
                                                    index_unique=None)
        reference_query_adata.obs['original_batch'] = reference_query_adata.obs[batch_key].tolist()

        # passed model as file
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            _validate_var_names(query, var_names)
            init_params = cls._get_init_params_from_dict(attr_dict)

            new_model = cls(reference_query_adata, **init_params)
            new_model.model.load_state_dict(model_state_dict)

            integrated_query = new_model.batch_removal(reference_query_adata, batch_key = "reference_map", cell_label_key = "cell_type", return_latent = True)

            return integrated_query

        #passed model as model object
        else:
            # when corrected_reference is already in the passed model
            if np.all(reference_model._get_user_attributes()[0][1].X == corrected_reference.X):
                integrated_query = reference_model.batch_removal(reference_query_adata, batch_key = "reference_map", cell_label_key = "cell_type", return_latent = True)
            else:
                attr_dict = reference_model._get_public_attributes()
                model_state_dict = reference_model.model.state_dict()
                init_params = cls._get_init_params_from_dict(attr_dict)

                new_model = cls(reference_query_adata, **init_params)
                new_model.model.load_state_dict(model_state_dict)

                integrated_query = new_model.batch_removal(reference_query_adata, batch_key = "reference_map", cell_label_key = "cell_type", return_latent = True)

            return integrated_query
