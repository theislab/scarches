import inspect
import os
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from typing import Optional, Union

from .trvae import trVAE
from scarches.trainers.trvae.unsupervised import trVAETrainer
from ._utils import _validate_var_names

class BaseMixin:
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


class TRVAE(BaseMixin):
    """Model for scArches class. This class contains the implementation of Conditional Variational Auto-encoder.

       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_mmd: Boolean
            If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
       mmd_on: String
            Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
            decoder layer.
       mmd_boundary: Integer or None
            Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
            conditions.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       beta: Float
            Scaling Factor for MMD loss
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """
    def __init__(
        self,
        adata: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = True,
        mmd_on: str = 'z',
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = 'nb',
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
    ):
        self.adata = adata

        self.condition_key_ = condition_key

        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.dr_rate_ = dr_rate
        self.use_mmd_ = use_mmd
        self.mmd_on_ = mmd_on
        self.mmd_boundary_ = mmd_boundary
        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln

        self.input_dim_ = adata.n_vars

        self.model = trVAE(
            self.input_dim_,
            self.conditions_,
            self.hidden_layer_sizes_,
            self.latent_dim_,
            self.dr_rate_,
            self.use_mmd_,
            self.mmd_on_,
            self.mmd_boundary_,
            self.recon_loss_,
            self.beta_,
            self.use_bn_,
            self.use_ln_,
        )

        self.is_trained_ = False

        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        **kwargs
    ):
        """Train the model.

           Parameters
           ----------
           n_epochs
                Number of epochs for training the model.
           lr
                Learning rate for training the model.
           eps
                torch.optim.Adam eps parameter
           kwargs
                kwargs for the TrVAE trainer.
        """
        self.trainer = trVAETrainer(
            self.model,
            self.adata,
            condition_key=self.condition_key_,
            **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        mean: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.

           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           mean
                return mean instead of random sample from the latent space

           Returns
           -------
                Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None:
                c = self.adata.obs[self.condition_key_]

        if c is not None:
            c = np.asarray(c)
            if not set(c).issubset(self.conditions_):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(c.shape[0])
            for condition, label in self.model.condition_encoder.items():
                labels[c == condition] = label
            c = torch.tensor(labels, device=device)

        x = torch.tensor(x, device=device)

        latents = []
        indices = torch.arange(x.size(0), device=device)
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(x[batch,:], c[batch], mean)
            latents += [latent.cpu().detach()]

        return np.array(torch.cat(latents))

    def get_y(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.

           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           Returns
           -------
                Returns array containing output of first decoder layer.
        """
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None:
                c = self.adata.obs[self.condition_key_]

        if c is not None:
            c = np.asarray(c)
            if not set(c).issubset(self.conditions_):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(c.shape[0])
            for condition, label in self.model.condition_encoder.items():
                labels[c == condition] = label
            c = torch.tensor(labels, device=device)

        x = torch.tensor(x, device=device)

        latents = []
        indices = torch.arange(x.size(0), device=device)
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_y(x[batch,:], c[batch])
            latents += [latent.cpu().detach()]

        return np.array(torch.cat(latents))

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_key': dct['condition_key_'],
            'conditions': dct['conditions_'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'latent_dim': dct['latent_dim_'],
            'dr_rate': dct['dr_rate_'],
            'use_mmd': dct['use_mmd_'],
            'mmd_on': dct['mmd_on_'],
            'mmd_boundary': dct['mmd_boundary_'],
            'recon_loss': dct['recon_loss_'],
            'beta': dct['beta_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct['input_dim_']:
            raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct['condition_key_']].unique().tolist()
        if not set(adata_conditions).issubset(dct['conditions_']):
            raise ValueError("Incorrect conditions")

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'TRVAE'],
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                TRVAE model to expand or a path to TRVAE model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.

           Returns
           -------
           new_model: trVAE
                New TRVAE model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
        init_params = cls._get_init_params_from_dict(attr_dict)

        conditions = init_params['conditions']
        condition_key = init_params['condition_key']

        new_conditions = []
        adata_conditions = adata.obs[condition_key].unique().tolist()
        # Check if new conditions are already known
        for item in adata_conditions:
            if item not in conditions:
                new_conditions.append(item)

        # Add new conditions to overall conditions
        for condition in new_conditions:
            conditions.append(condition)

        if remove_dropout:
            init_params['dr_rate'] = 0.0

        new_model = cls(adata, **init_params)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if freeze_expression:
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True

        return new_model
