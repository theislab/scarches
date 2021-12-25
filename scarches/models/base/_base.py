import inspect
import os
import torch
import pickle
import numpy as np

from copy import deepcopy
from anndata import AnnData, read
from typing import Optional, Union
from torch.distributions import Normal

from scarches.models.base._utils import _validate_var_names


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
                # only one dim diff
                new_shape = new_ten.shape
                n_dims = len(new_shape)
                sel = [slice(None)] * n_dims
                for i in range(n_dims):
                    dim_diff = new_shape[i] - load_ten.shape[i]
                    axs = i
                    sel[i] = slice(-dim_diff, None)
                    if dim_diff > 0:
                        break
                fixed_ten = torch.cat([load_ten, new_ten[tuple(sel)]], dim=axs)
                load_state_dict[key] = fixed_ten

        for key, ten in new_state_dict.items():
            if key not in load_state_dict:
                load_state_dict[key] = ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(cls, dir_path: str, map_location: Optional[str] = None):
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

        # Overwrite adata with new genes
        adata = _validate_var_names(adata, var_names)

        cls._validate_adata(adata, attr_dict)
        init_params = cls._get_init_params_from_dict(attr_dict)

        model = cls(adata, **init_params)
        model.model.load_state_dict(model_state_dict)
        model.model.eval()

        model.is_trained_ = attr_dict['is_trained_']

        return model


class SurgeryMixin:
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'Model'],
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        **kwargs
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                A model to expand or a path to a model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.

           Returns
           -------
           new_model
                New model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            adata = _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))

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

        init_params.update(kwargs)

        new_model = cls(adata, **init_params)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
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


class CVAELatentsMixin:
    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        mean: bool = False,
        mean_var: bool = False
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

        x = torch.tensor(x)

        latents = []
        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(x[batch,:].to(device), c[batch], mean, mean_var)
            latent = (latent,) if not isinstance(latent, tuple) else latent
            latents += [tuple(l.cpu().detach() for l in latent)]

        result = tuple(np.array(torch.cat(l)) for l in zip(*latents))
        result = result[0] if len(result) == 1 else result

        return result

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

        x = torch.tensor(x)

        latents = []
        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_y(x[batch,:].to(device), c[batch])
            latents += [latent.cpu().detach()]

        return np.array(torch.cat(latents))


class CVAELatentsModelMixin:
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.

           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.

           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False, mean_var=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean
           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        elif mean_var:
            return (z_mean, torch.exp(z_log_var) + 1e-4)
        return latent

    def get_y(self, x, c=None):
        """Map `x` in to the y dimension (First Layer of Decoder). This function will feed data in encoder  and return
           y for each sample in data.

           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.

           Returns
           -------
           Returns Torch Tensor containing output of first decoder layer.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        output = self.decoder(latent, c)
        return output[-1]
