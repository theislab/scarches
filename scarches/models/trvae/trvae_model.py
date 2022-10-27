import inspect
import os
from pathlib import PurePath
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union
from scipy.sparse import issparse

from .trvae import trVAE
from ...trainers.trvae.unsupervised import trVAETrainer
from ..base._utils import _validate_var_names
from ..base._base import BaseMixin, SurgeryMixin, CVAELatentsMixin


class TRVAE(BaseMixin, SurgeryMixin, CVAELatentsMixin):
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



 #TODO: Check for correctness
    def zero_shot_surgery(self, adata, model_path, force_cuda=False, copy=False, subsample=1.):
        # assert subsample > 0. and subsample <= 1.

        if copy:
            adata = adata.copy()

        with open(PurePath(model_path) / "attr.pkl", "rb") as handle:
            attr_dict = pickle.load(handle)

        ref_conditions = attr_dict["conditions_"]
        condition_key = attr_dict["condition_key_"]

        # if subsample < 1.:
        #     adata = subsample_conditions(adata, condition_key, subsample)

        original_key = "_original_" + condition_key
        adata.obs[original_key] = adata.obs[condition_key].copy()

        adata.strings_to_categoricals()

        original_cats = adata.obs[condition_key].unique()

        adata.obs[condition_key] = adata.obs[condition_key].cat.rename_categories(ref_conditions[:len(original_cats)])

        ref_model = self.load(model_path, adata)
        if force_cuda:
            ref_model.model = ref_model.model.cuda()

        device = next(ref_model.model.parameters()).device
        print("Device", device)

        rename_cats = {}

        for cat in original_cats:
            cat_mask = adata.obs[original_key] == cat
            X = adata.X[cat_mask]
            print("Processing original category:", cat, "n_obs:", X.shape[0])
            if issparse(X):
                X = X.toarray()
            X = torch.tensor(X, device=device)
            sizefactor = X.sum(-1)
            c = torch.zeros(X.shape[0], device=device, dtype=int)

            min_loss = None
            for ref_cat, ref_cat_val in ref_model.model.condition_encoder.items():
                print("  processing", ref_cat)
                c[:] = ref_cat_val
                recon_loss,_, _ = ref_model.model.forward(x=X, batch=c, sizefactor=sizefactor)
                if min_loss is None:
                    min_loss = recon_loss
                    rename_cats[cat] = ref_cat
                else:
                    if recon_loss < min_loss:
                        min_loss = recon_loss
                        rename_cats[cat] = ref_cat

        map_cats = adata.obs[original_key].map(rename_cats).astype("category")

        adata.obs[condition_key] = map_cats
        ref_model.adata.obs[condition_key] = map_cats

        return ref_model, rename_cats




#TODO: Check correctness
    def one_shot_surgery(
        self,
        adata,
        model_path,
        force_cuda=False,
        copy=False,
        subsample=1.,
        pretrain=1,
        **kwargs
    ):
        # assert subsample > 0. and subsample <= 1.

        if copy:
            adata = adata.copy()

        ref_model, rename_cats = self.zero_shot_surgery(adata, model_path, force_cuda=force_cuda, copy=False)

        cond_key = ref_model.condition_key_
        adata.obs[cond_key] = adata.obs["_original_" + cond_key]

        # if subsample < 1.:
        #     adata = subsample_conditions(adata, cond_key, subsample)

        query_model = self.load_query_data(adata, ref_model, **kwargs)

        cond_enc = query_model.model.condition_encoder

        to_set = [cond_enc[cat] for cat in rename_cats]
        to_get = [cond_enc[cat] for cat in rename_cats.values()]

        # query_model.model.embedding.weight.data[to_set] = query_model.model.embedding.weight.data[to_get]

        if pretrain > 0:
            query_model.train(n_epochs=pretrain, pretraining_epochs=pretrain)

        return query_model 




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


   
    