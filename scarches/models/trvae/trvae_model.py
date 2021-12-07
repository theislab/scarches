import inspect
import os
import torch
import pickle
import numpy as np
import pandas as pd

from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union

from .trvae import trVAE
from scarches.trainers.trvae.unsupervised import trVAETrainer
from scarches.trainers.trvae.regularized import VIATrainer
from ._utils import _validate_var_names

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
        map_location: Optional[str] = None
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

        attr_dict, model_state_dict, var_names = cls._load_params(dir_path, map_location=map_location)

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
        use_l_encoder: bool = False,
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
        mask: Optional[Union[np.ndarray, list]] = None,
        decoder_last_layer: str = "softmax",
        use_decoder_relu: bool = False,
        use_hsic: bool = False,
        n_ext_decoder: int = 0,
        n_ext_m_decoder: int = 0,
        n_expand_encoder: int = 0,
        soft_mask: bool = False,
        soft_ext_mask: bool = False,
        hsic_one_vs_all: bool = False,
        ext_mask: Optional[Union[np.ndarray, list]] = None
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

        self.use_decoder_relu_ = use_decoder_relu
        self.use_hsic_ = use_hsic and n_expand_encoder > 0
        self.hsic_one_vs_all_ = hsic_one_vs_all
        self.use_l_encoder_ = use_l_encoder
        self.decoder_last_layer_ = decoder_last_layer
        self.mask_ = None
        if mask is not None:
            self.mask_ = mask if isinstance(mask, list) else mask.tolist()
            mask = torch.tensor(mask).float()
            self.latent_dim_ = len(self.mask_)

        self.ext_mask_ = None
        if ext_mask is not None:
            self.ext_mask_ = ext_mask if isinstance(ext_mask, list) else ext_mask.tolist()
            ext_mask = torch.tensor(ext_mask).float()

        self.n_ext_decoder_ = n_ext_decoder
        self.n_expand_encoder_ = n_expand_encoder
        self.n_ext_m_decoder_ = n_ext_m_decoder

        self.soft_mask_ = soft_mask
        self.soft_ext_mask_ = soft_ext_mask

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
            self.use_l_encoder_,
            self.beta_,
            self.use_bn_,
            self.use_ln_,
            mask,
            self.decoder_last_layer_,
            self.use_decoder_relu_,
            self.use_hsic_,
            self.n_ext_decoder_,
            self.n_ext_m_decoder_,
            self.n_expand_encoder_,
            self.soft_mask_,
            self.soft_ext_mask_,
            self.hsic_one_vs_all_,
            ext_mask
        )

        self.is_trained_ = False

        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        alpha: Optional[float] = None,
        omega: Optional[torch.Tensor] = None,
        gamma_ext: Optional[float] = None,
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
        if self.mask_ is None:
            self.trainer = trVAETrainer(
                self.model,
                self.adata,
                condition_key=self.condition_key_,
                **kwargs)
            self.trainer.train(n_epochs, lr, eps)
        else:
            self.trainer = VIATrainer(
                self.model,
                self.adata,
                alpha=alpha,
                omega=omega,
                gamma_ext=gamma_ext,
                condition_key=self.condition_key_,
                **kwargs
            )
            self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

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

        is_mean_var = not mean and mean_var

        if is_mean_var:
            latents = [[], []]
        else:
            latents = []

        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(x[batch,:].to(device), c[batch], mean, mean_var)
            if is_mean_var:
                latents[0] += [latent[0].cpu().detach()]
                latents[1] += [latent[1].cpu().detach()]
            else:
                latents += [latent.cpu().detach()]

        if is_mean_var:
            merged = np.array(torch.cat(latents[0])), np.array(torch.cat(latents[1]))
        else:
            merged = np.array(torch.cat(latents))

        return merged

    def terms_genes(self, terms: Union[str, list]='terms'):
        if isinstance(terms, str):
            terms = self.adata.uns[terms]
        else:
            if len(terms) != len(self.mask_):
                raise ValueError('The list of terms should have the same length as the mask.')
        I = np.array(self.mask_, dtype=bool)
        return {term: self.adata.var_names[I[i]].tolist() for i, term in enumerate(terms)}

    def _latent_directions(self, method="sum", return_confidence=False):
        terms_weights = self.model.decoder.L0.expr_L.weight.data

        if method == "sum":
            signs = terms_weights.sum(0).cpu().numpy()
            signs[signs>0] = 1.
            signs[signs<0] = -1.
            confidence = None
        elif method == "counts":
            num_nz = torch.count_nonzero(terms_weights, dim=0)
            upreg_genes = torch.count_nonzero(terms_weights > 0, dim=0)
            signs = upreg_genes / (num_nz+(num_nz==0))
            signs = signs.cpu().numpy()

            confidence = signs.copy()
            confidence = np.abs(confidence-0.5)/0.5
            confidence[num_nz==0] = 0

            signs[signs>0.5] = 1.
            signs[signs<0.5] = -1.

            signs[signs==0.5] = 0
            signs[num_nz==0] = 0
        else:
            raise ValueError("Unrecognized method for getting the latent direction.")

        return signs if not return_confidence else (signs, confidence)

    def latent_enrich(
        self,
        groups,
        comparison="rest",
        n_perm=3000,
        directions=None,
        select_terms=None,
        adata=None,
        exact=False
    ):
        if adata is None:
            adata = self.adata

        if isinstance(groups, str):
            cats_col = adata.obs[groups]
            cats = cats_col.unique()
        elif isinstance(groups, dict):
            cats = []
            all_cells = []
            for group, cells in groups.items():
                cats.append(group)
                all_cells += cells
            adata = adata[all_cells]
            cats_col = pd.Series(index=adata.obs_names, dtype=str)
            for group, cells in groups.items():
                cats_col[cells] = group
        else:
            raise ValueError("groups should be a string or a dict.")

        if comparison != "rest" and set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            choice_1 = np.random.choice(cat_mask.sum(), n_perm)
            choice_2 = np.random.choice(others_mask.sum(), n_perm)

            adata_cat = adata[cat_mask][choice_1]
            adata_others = adata[others_mask][choice_2]

            z0 = self.get_latent(
                adata_cat.X,
                adata_cat.obs[self.condition_key_],
                mean=False,
                mean_var=exact
            )
            z1 = self.get_latent(
                adata_others.X,
                adata_others.obs[self.condition_key_],
                mean=False,
                mean_var=exact
            )

            if not exact:
                if directions is not None:
                    z0 *= directions
                    z1 *= directions

                if select_terms is not None:
                    z0 = z0[:, select_terms]
                    z1 = z1[:, select_terms]

                to_reduce = z0 > z1

                zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)
            else:
                from scipy.special import erfc

                means0, vars0 = z0
                means1, vars1 = z1

                if directions is not None:
                    means0 *= directions
                    means1 *= directions

                to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                to_reduce = 0.5 * erfc(to_reduce)

                zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            bf[zeros_mask] = 0

            scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

        return scores

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
            'mask': dct['mask_'],
            'use_decoder_relu': dct['use_decoder_relu_'],
            'use_hsic': dct['use_hsic_'] if 'use_hsic_' in dct else False,
            'decoder_last_layer': dct['decoder_last_layer_'] if 'decoder_last_layer_' in dct else "softmax",
            'use_l_encoder': dct['use_l_encoder_'] if 'use_l_encoder_' in dct else False,
            'n_ext_decoder': dct['n_ext_decoder_'] if 'n_ext_decoder_' in dct else 0,
            'n_ext_m_decoder': dct['n_ext_m_decoder_'] if 'n_ext_m_decoder_' in dct else 0,
            'n_expand_encoder': dct['n_expand_encoder_'] if 'n_expand_encoder_' in dct else 0,
            'soft_mask': dct['soft_mask_'] if 'soft_mask_' in dct else False,
            'soft_ext_mask': dct['soft_ext_mask_'] if 'soft_ext_mask_' in dct else False,
            'hsic_one_vs_all': dct['hsic_one_vs_all_'] if 'hsic_one_vs_all_' in dct else False,
            'ext_mask': dct['ext_mask_'] if 'ext_mask_' in dct else None
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
        unfreeze_ext: bool = True,
        remove_dropout: bool = True,
        new_n_ext_decoder: Optional[int] = None,
        new_n_ext_m_decoder: Optional[int] = None,
        new_n_expand_encoder: Optional[int] = None,
        new_ext_mask: Optional[Union[np.ndarray, list]] = None,
        new_soft_ext_mask: bool = False
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
        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))

        if new_n_ext_decoder is not None:
            init_params['n_ext_decoder'] = new_n_ext_decoder
        if new_n_expand_encoder is not None:
            init_params['n_expand_encoder'] = new_n_expand_encoder
        if new_n_ext_m_decoder is not None:
            init_params['n_ext_m_decoder'] = new_n_ext_m_decoder
            if new_ext_mask is None:
                raise ValueError('Provide new ext_mask')
            init_params['ext_mask'] = new_ext_mask
            init_params['soft_ext_mask'] = new_soft_ext_mask

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

                if unfreeze_ext:
                    if 'ext_L.weight' in name or 'ext_L_m.weight' in name:
                        p.requires_grad = True
                    if 'expand_mean_encoder' in name or 'expand_var_encoder' in name:
                        p.requires_grad = True

        return new_model
