import inspect
import os
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from typing import Optional, Union

from .vaearith import vaeArith
from .trainer import vaeArithTrainer
from scarches.models.trvae._utils import _validate_var_names

class BaseMixin:
    """ Adapted from
        Title: scvi-tools
        Authors: Romain Lopez <romain_lopez@gmail.com>,
                 Adam Gayoso <adamgayoso@berkeley.edu>,
                 Galen Xing <gx2113@columbia.edu>
        Date: 14.12.2020
        Code version: 0.8.0-beta.0
        Availability: https://github.com/YosefLab/scvi-tools
        Link to the used code:
        https://github.com/YosefLab/scvi-tools/blob/0.8.0-beta.0/scvi/core/models/base.py
    """
    def _get_user_attributes(self):
        # returns all the self attributes defined in a model class, eg, self.is_trained_
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.
           Neither the trainer optimizer state nor the trainer history are saved.
           Parameters
           ----------
           dir_path
                Path to a directory.
           overwrite
                Overwrite existing data or not. If `False` and directory
                already exists at `dir_path`, error will be raised.
           save_anndata
                If True, also saves the anndata
           anndata_write_kwargs
                Kwargs for anndata write function
        """
        # get all the public attributes
        public_attributes = self._get_public_attributes()
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        if save_anndata:
            self.adata.write(
                os.path.join(dir_path, "adata.h5ad"), **anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.model.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(
        cls,
        dir_path: str,
        map_location: Optional[str] = None
    ):
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        model_path = os.path.join(dir_path, "model_params.pt")
        varnames_path = os.path.join(dir_path, "var_names.csv")

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        model_state_dict = torch.load(model_path, map_location=map_location)

        var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

        return attr_dict, model_state_dict, var_names

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
    ):
        """Instantiate a model from the saved output.
           Parameters
           ----------
           dir_path
                Path to saved outputs.
           adata
                AnnData object.
                If None, will check for and load anndata saved with the model.
           Returns
           -------
                Model with loaded state dictionaries.
        """
        adata_path = os.path.join(dir_path, "adata.h5ad")

        load_adata = adata is None

        if os.path.exists(adata_path) and load_adata:
            adata = read(adata_path)
        elif not os.path.exists(adata_path) and load_adata:
            raise ValueError("Save path contains no saved anndata and no adata was passed.")

        attr_dict, model_state_dict, var_names = cls._load_params(dir_path)

        _validate_var_names(adata, var_names)
        cls._validate_adata(adata, attr_dict)
        init_params = cls._get_init_params_from_dict(attr_dict)

        model = cls(adata, **init_params)
        model.model.load_state_dict(model_state_dict)
        model.model.eval()

        model.is_trained_ = attr_dict['is_trained_']

        return model


class SCGEN(BaseMixin):
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

    def train(self, n_epochs = 25, batch_size = 32):
        self.trainer = vaeArithTrainer(self.model, self.adata, n_epochs, batch_size)
        self.trainer.train()
        self.is_trained_ = True


    def to_latent(self, data):
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
        data = torch.tensor(data, device=device) # to tensor
        latent = self.model.to_latent(data)
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


    def batch_removal(self, adata, batch_key="batch", cell_label_key="cell_type"):
        """
    Removes batch effect of adata

    Parameters
    ----------
    network: `scgen VAE`
        Variational Auto-encoder class object after training the network.
    adata: `~anndata.AnnData`
        Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
    batch_key: `str`
        batch label key in  adata.obs
    cell_label_key: `str`
        cell type label key in adata.obs
    return_latent: `bool`
        if `True` the returns corrected latent representation

    Returns
    -------
    corrected: `~anndata.AnnData`
        adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent"].
     """
        if sparse.issparse(adata.X):
        latent_all = network.to_latent(adata.X.A)
    else:
        latent_all = network.to_latent(adata.X)
    adata_latent = anndata.AnnData(latent_all)
    adata_latent.obs = adata.obs.copy(deep=True)
    unique_cell_types = np.unique(adata_latent.obs[cell_label_key])
    shared_ct = []
    not_shared_ct = []
    for cell_type in unique_cell_types:
        temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
        if len(np.unique(temp_cell.obs[batch_key])) < 2:
            cell_type_ann = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
            not_shared_ct.append(cell_type_ann)
            continue
        temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
        batch_list = {}
        batch_ind = {}
        max_batch = 0
        max_batch_ind = ""
        batches = np.unique(temp_cell.obs[batch_key])
        for i in batches:
            temp = temp_cell[temp_cell.obs[batch_key] == i]
            temp_ind = temp_cell.obs[batch_key] == i
            if max_batch < len(temp):
                max_batch = len(temp)
                max_batch_ind = i
            batch_list[i] = temp
            batch_ind[i] = temp_ind
        max_batch_ann = batch_list[max_batch_ind]
        for study in batch_list:
            delta = np.average(max_batch_ann.X, axis=0) - np.average(batch_list[study].X, axis=0)
            batch_list[study].X = delta + batch_list[study].X
            temp_cell[batch_ind[study]].X = batch_list[study].X
        shared_ct.append(temp_cell)
    all_shared_ann = anndata.AnnData.concatenate(*shared_ct, batch_key="concat_batch", index_unique=None)
    if "concat_batch" in all_shared_ann.obs.columns:
        del all_shared_ann.obs["concat_batch"]
    if len(not_shared_ct) < 1:
        corrected = sc.AnnData(network.reconstruct(all_shared_ann.X, use_data=True),obs=all_shared_ann.obs)
        corrected.var_names = adata.var_names.tolist()
        corrected = corrected[adata.obs_names]
        if adata.raw is not None:
            adata_raw = anndata.AnnData(X=adata.raw.X, var=adata.raw.var)
            adata_raw.obs_names = adata.obs_names
            corrected.raw = adata_raw
        corrected.obsm["latent"] = all_shared_ann.X
        return corrected
    else:
        all_not_shared_ann = anndata.AnnData.concatenate(*not_shared_ct, batch_key="concat_batch", index_unique=None)
        all_corrected_data = anndata.AnnData.concatenate(all_shared_ann, all_not_shared_ann, batch_key="concat_batch", index_unique=None)
        if "concat_batch" in all_shared_ann.obs.columns:
            del all_corrected_data.obs["concat_batch"]
        corrected = sc.AnnData( network.reconstruct(all_corrected_data.X, use_data=True), all_corrected_data.obs)
        corrected.var_names = adata.var_names.tolist()
        corrected = corrected[adata.obs_names]
        if adata.raw is not None:
            adata_raw = anndata.AnnData(X=adata.raw.X, var=adata.raw.var)
            adata_raw.obs_names = adata.obs_names
            corrected.raw = adata_raw
        corrected.obsm["latent"] = all_corrected_data.X
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
    def load_query_data(cls,
                      adata: AnnData,
                      reference_model: Union[str, 'SCGEN'],
                      freeze: bool = True,
                      freeze_expression: bool = True,
                      remove_dropout: bool = True):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.
           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                SCGEN model to expand or a path to SCGEN model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.
           Returns
           -------
           new_model: scgen
                New SCGEN model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
        init_params = cls._get_init_params_from_dict(attr_dict)


        new_model = cls(adata, **init_params)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if "L0" in name or "N0" in name:
                    p.requires_grad = True

        return new_model
