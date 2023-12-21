from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from collections import defaultdict
from scipy import sparse
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from copy import deepcopy

from ..base._base import BaseMixin
from ..base._utils import _validate_var_names
from .scpoli import scpoli
from ...trainers.scpoli import scPoliTrainer
from ...trainers.scpoli.trainer import custom_collate
from ...dataset import MultiConditionAnnotatedDataset


class scPoli(BaseMixin):
    """Model for scPoli class. This class contains the methods and functionalities for label transfer and prototype training.

    Parameters
    ----------
    adata: : `~anndata.AnnData`
        Annotated data matrix.
    share_metadata : Bool
        Whether or not to share metadata associated with samples. The metadata is aggregated using the condition_keys. First element is
        taken. Consider manually adding an .obs_metadata attribute if you need more flexibility.
    condition_keys: String
        column name of conditions in `adata.obs` data frame.
    conditions: List
        List of Condition names that the used data will contain to get the right encoding when used after reloading.
    cell_type_keys: List or str
        List or string of obs columns to use as cell type annotation for prototypes.
    cell_types: Dictionary
        Dictionary of cell types. Keys are cell types and values are cell_type_keys. Needed for surgery.
    unknown_ct_names: List
        List of strings with the names of cell clusters to be ignored for prototypes computation.
    labeled_indices: List
        List of integers with the indices of the labeled cells.
    prototypes_labeled: Dictionary
        Dictionary with keys mean, cov and the respective mean or covariate matrices for prototypes.
    prototypes_unlabeled: Dictionary
        Dictionary with keys mean and the respective mean for unlabeled prototypes.
    hidden_layer_sizes: List
        A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
    latent_dim: Integer
        Bottleneck layer (z)  size.
    embedding_dim: Integer
        Conditional embedding size.
    embedding_max_norm:
        Max norm allowed for conditional embeddings.
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
        share_metadata: Optional[bool] = True,
        obs_metadata: Optional[pd.DataFrame] = None,
        condition_keys: Optional[Union[list, str]] = None,
        conditions: Optional[list] = None,
        conditions_combined: Optional[list] = None,
        inject_condition: Optional[list] = ["encoder", "decoder"],
        cell_type_keys: Optional[Union[str, list]] = None,
        cell_types: Optional[dict] = None,
        unknown_ct_names: Optional[list] = None,
        labeled_indices: Optional[list] = None,
        prototypes_labeled: Optional[dict] = None,
        prototypes_unlabeled: Optional[dict] = None,
        hidden_layer_sizes: list = None,
        latent_dim: int = 10,
        embedding_dims: Union[list, int] = 10,
        embedding_max_norm: float = 1.0,
        dr_rate: float = 0.05,
        use_mmd: bool = False,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = "nb",
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
    ):
        # gather data information
        self.adata = adata
        self.share_metadata_ = share_metadata

        if isinstance(condition_keys, str):
            self.condition_keys_ = [condition_keys]
        else:
            self.condition_keys_ = condition_keys


        if isinstance(cell_type_keys, str):
            self.cell_type_keys_ = [cell_type_keys]
        else:
            self.cell_type_keys_ = cell_type_keys

        if unknown_ct_names is not None and type(unknown_ct_names) is not list:
            raise TypeError(
                f"Parameter 'unknown_ct_names' has to be list not {type(unknown_ct_names)}"
            )
        self.unknown_ct_names_ = unknown_ct_names

        if labeled_indices is None:
            self.labeled_indices_ = range(len(adata))
        else:
            self.labeled_indices_ = labeled_indices

        if conditions is None:
            if condition_keys is not None:
                self.conditions_ = {}
                for cond in self.condition_keys_:
                    self.conditions_[cond] = adata.obs[cond].unique().tolist()
            else:
                self.conditions_ = {}
        else:
            self.conditions_ = conditions

        if conditions_combined is None:
            if len(self.condition_keys_) > 1:
                self.adata.obs['conditions_combined'] = adata.obs[condition_keys].apply(lambda x: '_'.join(x), axis=1)
            else:
                self.adata.obs['conditions_combined'] = adata.obs[condition_keys]
            self.conditions_combined_ = self.adata.obs['conditions_combined'].unique().tolist()
        else:
            self.conditions_combined_ = conditions_combined

        if obs_metadata is not None:
            self.obs_metadata_ = obs_metadata
        elif self.share_metadata_ is True:
            self.obs_metadata_ = adata.obs.groupby('conditions_combined').first()
        else:
            self.obs_metadata_ = []

        if self.share_metadata_:
            self.obs_metadata_ = adata.obs.groupby(condition_keys).first()

        # Gather all cell type information
        if cell_types is None:
            if cell_type_keys is not None:
                self.cell_types_ = dict()
                for cell_type_key in self.cell_type_keys_:
                    uniq_cts = (
                        adata.obs[cell_type_key][self.labeled_indices_]
                        .unique()
                        .tolist()
                    )
                    for ct in uniq_cts:
                        if ct in self.cell_types_:
                            self.cell_types_[ct].append(cell_type_key)
                        else:
                            self.cell_types_[ct] = [cell_type_key]
            else:
                self.cell_types_ = dict()
        else:
            self.cell_types_ = cell_types

        if self.unknown_ct_names_ is not None:
            for unknown_ct in self.unknown_ct_names_:
                if unknown_ct in self.cell_types_:
                    del self.cell_types_[unknown_ct]


        # store model parameters
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes_ = [int(np.ceil(np.sqrt(adata.shape[1])))]
        else:
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
        self.inject_condition_ = inject_condition
        if isinstance(embedding_dims, int):
            self.embedding_dims_ = [embedding_dims] * len(self.condition_keys_)
        else:
            assert len(embedding_dims) == len(self.condition_keys_), \
                "Embedding dimensions passed do not match condition keys"
            self.embedding_dims_ = embedding_dims
        self.embedding_max_norm_ = embedding_max_norm

        self.input_dim_ = adata.n_vars
        self.prototypes_labeled_ = (
            {"mean": None, "cov": None}
            if prototypes_labeled is None
            else prototypes_labeled
        )
        self.prototypes_unlabeled_ = (
            {
                "mean": None,
            }
            if prototypes_unlabeled is None
            else prototypes_unlabeled
        )

        self.model_cell_types = list(self.cell_types_.keys())
        self.is_trained_ = False
        self.trainer = None

        self.model = scpoli(
            input_dim=self.input_dim_,
            conditions=self.conditions_,
            conditions_combined=self.conditions_combined_,
            cell_types=self.model_cell_types,
            inject_condition=self.inject_condition_,
            embedding_dims=self.embedding_dims_,
            embedding_max_norm=self.embedding_max_norm_,
            unknown_ct_names=self.unknown_ct_names_,
            prototypes_labeled=self.prototypes_labeled_,
            prototypes_unlabeled=self.prototypes_unlabeled_,
            hidden_layer_sizes=self.hidden_layer_sizes_,
            latent_dim=self.latent_dim_,
            dr_rate=self.dr_rate_,
            recon_loss=self.recon_loss_,
            beta=self.beta_,
            use_bn=self.use_bn_,
            use_ln=self.use_ln_,
        )

        if self.prototypes_labeled_["mean"] is not None:
            self.prototypes_labeled_["mean"] = self.prototypes_labeled_["mean"].to(
                next(self.model.parameters()).device
            )
            self.prototypes_labeled_["cov"] = self.prototypes_labeled_["cov"].to(
                next(self.model.parameters()).device
            )
        if self.prototypes_unlabeled_["mean"] is not None:
            self.prototypes_unlabeled_["mean"] = self.prototypes_unlabeled_["mean"].to(
                next(self.model.parameters()).device
            )

    def train(
        self,
        n_epochs: int = 100,
        pretraining_epochs=None,
        eta: float = 1,
        lr: float = 1e-3,
        eps: float = 0.01,
        alpha_epoch_anneal = 1e2,
        reload_best: bool = False,
        prototype_training: Optional[bool] = True,
        unlabeled_prototype_training: Optional[bool] = True,
        **kwargs,
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
             kwargs for the scPoli trainer.
        """
        self.prototype_training_ = prototype_training
        self.unlabeled_prototype_training_ = unlabeled_prototype_training
        if self.cell_type_keys_ is None:
            pretraining_epochs = n_epochs
            self.prototype_training_ = False
            print("The model is being trained without using prototypes.")
        elif pretraining_epochs is None:
            pretraining_epochs = int(np.floor(n_epochs * 0.9))


        self.trainer = scPoliTrainer(
            self.model,
            self.adata,
            labeled_indices=self.labeled_indices_,
            pretraining_epochs=pretraining_epochs,
            condition_keys=self.condition_keys_,
            cell_type_keys=self.cell_type_keys_,
            reload_best=reload_best,
            prototype_training=self.prototype_training_,
            unlabeled_prototype_training=self.unlabeled_prototype_training_,
            eta=eta,
            alpha_epoch_anneal=alpha_epoch_anneal,
            **kwargs,
        )
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
        self.prototypes_labeled_ = self.model.prototypes_labeled
        self.prototypes_unlabeled_ = self.model.prototypes_unlabeled

    def get_latent(
        self,
        adata,
        mean: bool = False,
        mean_var: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.

            Parameters
            ----------
            x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
            c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
            mean
                return mean instead of random sample from the latent space

        Returns
        -------
                Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        x = adata.X
        c = {k: adata.obs[k].values for k in self.condition_keys_}

        if isinstance(c, dict):
            label_tensor = []
            for cond in c.keys():
                query_conditions = c[cond]
                if not set(query_conditions).issubset(self.conditions_[cond]):
                    raise ValueError("Incorrect conditions")
                labels = np.zeros(query_conditions.shape[0])
                for condition, label in self.model.condition_encoders[cond].items():
                    labels[query_conditions == condition] = label
                label_tensor.append(labels)
            c = torch.tensor(label_tensor, device=device).T

        latents = []
        # batch the latent transformation process
        indices = torch.arange(x.shape[0])
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            x_batch = x[batch, :]
            if sparse.issparse(x_batch):
                x_batch = x_batch.toarray()
            x_batch = torch.tensor(x_batch, device=device)
            latent = self.model.get_latent(
                x_batch, c[batch, :], mean
            )
            latents += [latent.cpu().detach()]
        latents = torch.cat(latents)
        return np.array(latents)

    def get_conditional_embeddings(self):
        """
        Returns anndata object of the conditional embeddings
        """
        embeddings = [self.model.embeddings[i].weight.cpu().detach().numpy() for i in range(len(self.model.embeddings))]
        adata_emb = {}
        for i, cond in enumerate(self.conditions_.keys()):
            adata_emb[cond] =  sc.AnnData(
                X=embeddings[i],
                obs=pd.DataFrame(index=self.conditions_[cond])
            )
        #if self.share_metadata_:
        #    adata_emb.obs = self.obs_metadata_
        if len(self.condition_keys_) == 1:
            return adata_emb[self.condition_keys_[0]]
        else:
            return adata_emb

    #def get_combined_conditional_embeddings(self):
    #    """
    #    Returns anndata object of the conditional embeddings
    #    """
    #    embeddings = [self.model.embeddings[i].weight.cpu().detach().numpy() for i in range(len(self.model.embeddings))]
    #    adata_emb = {}
    #    for i, cond in enumerate(self.conditions_.keys()):
    #        adata_emb[cond] =  sc.AnnData(
    #            X=embeddings[i],
    #            obs=pd.DataFrame(index=self.conditions_[cond])
    #        )
    #    unique_conditions = self.adata.obs[self.condition_keys_].drop_duplicates()
    #    combined_embeddings = []
    #    for i in range(len(unique_conditions)):
    #        embs = []
    #        for cond in self.condition_keys_:
    #            embs.append(np.squeeze(adata_emb[cond][adata_emb[cond].obs_names == unique_conditions.iloc[i][cond]].X))
    #        embs = np.hstack(embs)
    #        combined_embeddings.append(embs)
    #    adata_emb_combined = sc.AnnData(
    #            X=np.vstack(combined_embeddings),
    #            obs=unique_conditions
    #        )


        #if self.share_metadata_:
        #    adata_emb.obs = self.obs_metadata_
    #    return adata_emb_combined

    def classify(
        self,
        adata,
        prototype=False,
        p=2,
        get_prob=False,
        log_distance=True,
        scale_uncertainties=False,
    ):
        """
        Classifies unlabeled cells using the prototypes obtained during training.
        Data handling before call to model's classify method.

        x:  np.ndarray
            Features to be classified. If None the stored
            model's adata is used.
        c: Dict or np.ndarray
            Condition vector, or dictionary when the model is conditioned on multiple
            batch covariates.
        prototype:
            Boolean whether to classify the gene features or prototypes stored
            stored in the model.

        """

        assert self.prototypes_labeled_['mean'] is not None, f"Model was trained without prototypes"

        device = next(self.model.parameters()).device
        self.model.eval()
        if prototype is False:
            x = adata.X
            c = {k: adata.obs[k].values for k in self.condition_keys_}
            if isinstance(c, dict):
                label_tensor = []
                for cond in c.keys():
                    query_conditions = c[cond]
                    if not set(query_conditions).issubset(self.conditions_[cond]):
                        raise ValueError("Incorrect conditions")
                    labels = np.zeros(query_conditions.shape[0])
                    for condition, label in self.model.condition_encoders[cond].items():
                        labels[query_conditions == condition] = label
                    label_tensor.append(labels)
                c = torch.tensor(label_tensor, device=device).T
        else:
            x = adata

        if sparse.issparse(x):
            x = x.A
        x = torch.tensor(x, device=device)

        results = dict()
        # loop through hierarchies
        for cell_type_key in self.cell_type_keys_:
            prototypes_idx = list()
            # get indices of different prototypes corresponding to current hierarchy
            for i, key in enumerate(self.cell_types_.keys()):
                if cell_type_key in self.cell_types_[key]:
                    prototypes_idx.append(i)

            prototypes_idx = torch.tensor(prototypes_idx, device=device)

            preds = []
            uncert = []
            weighted_distances = []
            indices = torch.arange(x.size(0), device=device)
            subsampled_indices = indices.split(512)
            for batch in subsampled_indices:
                if prototype:  # classify prototypes used for unseen cell type
                    pred, prob, weighted_distance = self.model.classify(
                        x[batch, :].to(device),
                        prototype=prototype,
                        classes_list=prototypes_idx,
                        p=p,
                        get_prob=get_prob,
                        log_distance=log_distance,
                    )
                else:  # default routine, classify cell by cell
                    pred, prob, weighted_distance = self.model.classify(
                        x[batch, :].to(device),
                        c[batch].to(device),
                        prototype=prototype,
                        classes_list=prototypes_idx,
                        p=p,
                        get_prob=get_prob,
                        log_distance=log_distance,
                    )
                preds += [pred.cpu().detach()]
                uncert += [prob.cpu().detach()]
                weighted_distances += [weighted_distance.cpu().detach()]

            full_pred = np.array(torch.cat(preds))
            full_uncert = np.array(torch.cat(uncert))
            full_weighted_distances = np.array(torch.cat(weighted_distances))
            inv_ct_encoder = {v: k for k, v in self.model.cell_type_encoder.items()}
            full_pred_names = []

            for _, pred in enumerate(full_pred):
                full_pred_names.append(inv_ct_encoder[pred])

            if scale_uncertainties is True:
                full_uncert = RobustScaler().fit_transform(full_uncert.reshape(-1, 1))
                full_uncert = MinMaxScaler(feature_range=(0, 1)).fit_transform(full_uncert).reshape(-1)

            results[cell_type_key] = {
                "preds": np.array(full_pred_names),
                "uncert": full_uncert,
                "weighted_distances": full_weighted_distances,
            }
        return results

    def add_new_cell_type(
        self,
        cell_type_name,
        obs_key,
        prototypes,
        x=None,
        c=None,
    ):
        """
        Function used to add new annotation for a novel cell type.

        Parameters
        ----------
        cell_type_name: str
            Name of the new cell type
        obs_key: str
            Obs column key to define the hierarchy level of celltype annotation.
        prototypes: list
            List of indices of the unlabeled prototypes that correspond to the new cell type
        x:  np.ndarray
            Features to be classified. If None the stored
            model's adata is used.
        c: np.ndarray
            Condition vector. If None the stored
            model's condition vector is used.

        Returns
        -------

        """
        # Get latent of model data or input data
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None:
                c = {cond: self.adata.obs[cond].values for cond in self.condition_keys_}

        if c is not None:
            label_tensor = []
            for cond in c.keys():
                query_conditions = c[cond]
                if not set(query_conditions).issubset(self.conditions_[cond]):
                    raise ValueError("Incorrect conditions")
                labels = np.zeros(query_conditions.shape[0])
                for condition, label in self.model.condition_encoders[cond].items():
                    labels[query_conditions == condition] = label
                label_tensor.append(labels)
            c = torch.tensor(label_tensor, device=device).T

        if sparse.issparse(x):
            x = x.A
        x = torch.tensor(x, device=device)
        latents = []
        indices = torch.arange(x.size(0), device=device)
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                x[batch, :].to(device), c[batch].to(device), False
            )
            latents += [latent.cpu().detach()]
        latents = torch.cat(latents)

        # get indices of different prototypes corresponding to current hierarchy
        prototypes_idx = list()
        for i, key in enumerate(self.cell_types_.keys()):
            if obs_key in self.cell_types_[key]:
                prototypes_idx.append(i)

        prototypes_idx = torch.tensor(prototypes_idx, device=device)

        # Calculate mean and Cov of new prototype
        self.model.add_new_cell_type(
            latents,
            cell_type_name,
            prototypes,
            prototypes_idx,
        )

        # Update parameters
        self.prototypes_labeled_ = self.model.prototypes_labeled
        self.prototypes_unlabeled_ = self.model.prototypes_unlabeled
        self.cell_types_[cell_type_name] = [obs_key]

    def get_prototypes_info(
        self,
        prototype_set="labeled",
    ):
        """
        Generates anndata file with prototype features and annotations.

        Parameters
        ----------
        cell_type_name: str
            Name of the new cell type
        prototypes: list
            List of indices of the unlabeled prototypes that correspond to the new cell type

        Returns
        -------

        """
        if prototype_set == "labeled":
            prototypes = self.prototypes_labeled_["mean"].detach().cpu().numpy()
            batch_name = "prototype-Set Labeled"
        elif prototype_set == "unlabeled":
            prototypes = self.prototypes_unlabeled_["mean"].detach().cpu().numpy()
            batch_name = "prototype-Set Unlabeled"
        else:
            print(
                f"Parameter 'prototype_set' has either to be 'labeled' for labeled prototype set or 'unlabeled' "
                f"for the unlabeled prototype set. But given value was {prototype_set}"
            )
            return
        prototypes_info = sc.AnnData(prototypes)
        prototypes_info.obs['batch'] = np.array(
            (prototypes.shape[0] * [batch_name])
        )

        results = self.classify(
            prototypes,
            prototype=True,
        )
        for cell_type_key in self.cell_type_keys_:
            if prototype_set == "l":
                truth_names = list()
                for key in self.cell_types_.keys():
                    if cell_type_key in self.cell_types_[key]:
                        truth_names.append(key)
                    else:
                        truth_names.append("nan")
            else:
                truth_names = list()
                for i in range(prototypes.shape[0]):
                    truth_names.append(f"{i}")

            prototypes_info.obs[cell_type_key] = np.array(truth_names)
            prototypes_info.obs[cell_type_key + "_pred"] = results[cell_type_key][
                "preds"
            ]
            prototypes_info.obs[cell_type_key + "_uncert"] = results[cell_type_key][
                "uncert"
            ]
        return prototypes_info

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct["input_dim_"]:
            raise ValueError("Incorrect var dimension")

        #adata_conditions = adata.obs[dct["condition_key_"]].unique().tolist()
        #if not set(adata_conditions).issubset(dct["conditions_"]):
        #    raise ValueError("Incorrect conditions")

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            "share_metadata": dct["share_metadata_"],
            "obs_metadata": dct["obs_metadata_"],
            "condition_keys": dct["condition_keys_"],
            "conditions": dct["conditions_"],
            "conditions_combined": dct["conditions_combined_"],
            "cell_type_keys": dct["cell_type_keys_"],
            "cell_types": dct["cell_types_"],
            "labeled_indices": dct["labeled_indices_"],
            "prototypes_labeled": dct["prototypes_labeled_"],
            "prototypes_unlabeled": dct["prototypes_unlabeled_"],
            #"prototype_training": dct["prototype_training_"],
            #"unlabeled_prototype_training": dct["unlabeled_prototype_training_"],
            "hidden_layer_sizes": dct["hidden_layer_sizes_"],
            "latent_dim": dct["latent_dim_"],
            "dr_rate": dct["dr_rate_"],
            "use_mmd": dct["use_mmd_"],
            "mmd_on": dct["mmd_on_"],
            "mmd_boundary": dct["mmd_boundary_"],
            "recon_loss": dct["recon_loss_"],
            "beta": dct["beta_"],
            "use_bn": dct["use_bn_"],
            "use_ln": dct["use_ln_"],
            "embedding_dims": dct["embedding_dims_"],
            "embedding_max_norm": dct["embedding_max_norm_"],
            "inject_condition": dct["inject_condition_"],
        }

        return init_params

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, "SCPOLI"],
        labeled_indices: Optional[list] = None,
        unknown_ct_names: Optional[list] = None,
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        return_new_conditions: bool = False,
        map_location = None,
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

        Parameters
        ----------
        adata
             Query anndata object.
        reference_model
             SCPOLI model to expand or a path to SCPOLI model folder.
        labeled_indices: List
             List of integers with the indices of the labeled cells.
        unknown_ct_names: List
             List of strings with the names of cell clusters to be ignored for prototypes computation.
        freeze: Boolean
             If 'True' freezes every part of the network except the first layers of encoder/decoder.
        freeze_expression: Boolean
             If 'True' freeze every weight in first layers except the condition weights.
        remove_dropout: Boolean
             If 'True' remove Dropout for Transfer Learning.
        map_location
             map_location to remap storage locations (as in '.load') of 'reference_model'.
             Only taken into account if 'reference_model' is a path to a model on disk.
        Returns
        -------
        new_model: scPoli
             New SCPOLI model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model, map_location)
            adata = _validate_var_names(adata, var_names)
        else:
            attr_dict = deepcopy(reference_model._get_public_attributes())
            model_state_dict = reference_model.model.state_dict()
        init_params = cls._get_init_params_from_dict(attr_dict)

        conditions = init_params["conditions"]
        n_reference_conditions = len(conditions)
        condition_keys = init_params["condition_keys"]

        new_conditions = defaultdict(list)
        adata_conditions = adata.obs[condition_keys].drop_duplicates()
        # Check if new conditions are already known
        for cond in condition_keys:
            unique_conditions = adata_conditions[cond].unique()
            for item in unique_conditions:
                if item not in conditions[cond]:
                    new_conditions[cond].append(item)

        # Add new conditions to overall conditions

        for cond in condition_keys:
            for condition in new_conditions[cond]:
                conditions[cond].append(condition)

        conditions_combined = init_params["conditions_combined"]
        if len(condition_keys) > 1:
            adata.obs['conditions_combined'] = adata.obs[condition_keys].apply(lambda x: '_'.join(x), axis=1)
        else:
            adata.obs['conditions_combined'] = adata.obs[condition_keys]
        new_conditions_combined = adata.obs['conditions_combined'].unique().tolist()
        for item in new_conditions_combined:
            if item not in conditions_combined:
                conditions_combined.append(item)

        obs_metadata = attr_dict["obs_metadata_"]
        new_obs_metadata = adata.obs.groupby('conditions_combined').first()
        obs_metadata = pd.concat([obs_metadata, new_obs_metadata])
        init_params["obs_metadata"] = obs_metadata

        cell_types = init_params["cell_types"]
        cell_type_keys = init_params["cell_type_keys"]
        # Check for cell types in new adata
        if cell_type_keys is not None:
            adata_cell_types = dict()
            for cell_type_key in cell_type_keys:
                uniq_cts = adata.obs[cell_type_key][labeled_indices].unique().tolist()
                for ct in uniq_cts:
                    if ct in adata_cell_types:
                        adata_cell_types[ct].append(cell_type_key)
                    else:
                        adata_cell_types[ct] = [cell_type_key]

            if unknown_ct_names is not None:
                for unknown_ct in unknown_ct_names:
                    if unknown_ct in adata_cell_types:
                        del adata_cell_types[unknown_ct]

            # Check if new conditions are already known and if not add them
            for key in adata_cell_types:
                if key not in cell_types:
                    cell_types[key] = adata_cell_types[key]

        if remove_dropout:
            init_params["dr_rate"] = 0.0

        init_params["labeled_indices"] = labeled_indices
        init_params["unknown_ct_names"] = unknown_ct_names
        new_model = cls(adata, **init_params)
        new_model.model.n_reference_conditions = n_reference_conditions
        #new_model.obs_metadata_ = obs_metadata
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if "embedding" in name:
                    p.requires_grad = True
                if "theta" in name:
                    p.requires_grad = True
                if freeze_expression:
                    if "cond_L.weight" in name:
                        p.requires_grad = False
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = False

        if return_new_conditions:
            return new_model, new_conditions
        else:
            return new_model

    @classmethod
    def shot_surgery(
        cls,
        adata: AnnData,
        reference_model: Union[str, "SCPOLI"],
        labeled_indices: Optional[list] = None,
        unknown_ct_names: Optional[list] = None,
        train_epochs: int = 0,
        batch_size: int = 128,
        subsample: float = 1.,
        force_cuda: bool = True,
        **kwargs
    ):
        assert subsample > 0. and subsample <= 1.

        model, new_conditions = cls.load_query_data(
            adata,
            reference_model,
            labeled_indices,
            unknown_ct_names,
            freeze = train_epochs>0,
            return_new_conditions=True
        )

        assert len(model.condition_keys_) == 1

        if force_cuda and torch.cuda.is_available():
            model.model.cuda()

        model.model.eval()

        cond_key = model.condition_keys_[0]
        new_conditions_list = new_conditions[cond_key]

        mapping = {}
        for new_cond in new_conditions_list:
            print(f"Processing {new_cond}.")
            adata_cond = adata[adata.obs[cond_key] == new_cond]
            if subsample < 1.:
                n_obs = len(adata_cond)
                n_ss = int(subsample * n_obs)
                idx = np.random.choice(n_obs, n_ss, replace=False)
                adata_cond = adata_cond[idx]

            min_recon_loss = None
            for old_cond in model.conditions_[cond_key]:
                if old_cond in new_conditions_list:
                    continue
                condition_encoders = deepcopy(model.model.condition_encoders)
                conditions_combined_encoder = deepcopy(model.model.conditions_combined_encoder)
                condition_encoders[cond_key][new_cond] = condition_encoders[cond_key][old_cond]
                conditions_combined_encoder[new_cond] = conditions_combined_encoder[old_cond]

                recon_loss = model.get_recon_loss(
                    adata_cond,
                    batch_size,
                    condition_encoders,
                    conditions_combined_encoder
                )
                print(f"  Using {old_cond}, recon loss {recon_loss}.")
                if min_recon_loss is None or recon_loss < min_recon_loss:
                    min_recon_loss = recon_loss
                    mapping[new_cond] = old_cond

        condition_encoder = model.model.condition_encoders[cond_key]
        conditions_combined_encoder = model.model.conditions_combined_encoder
        embeddings = model.model.embeddings[0].weight.data
        theta = None if model.model.theta is None else model.model.theta.data
        for new_cond in mapping:
            embeddings[condition_encoder[new_cond]] = embeddings[condition_encoder[mapping[new_cond]]]
            if theta is not None:
                theta[:, conditions_combined_encoder[new_cond]] = theta[:, conditions_combined_encoder[mapping[new_cond]]]

        if train_epochs > 0:
            model.train(n_epochs=train_epochs, **kwargs)

        return model, mapping

    def get_recon_loss(
        self,
        adata: AnnData,
        batch_size: int = 128,
        condition_encoders: Optional[dict] = None,
        conditions_combined_encoder: Optional[dict] = None
    ):
        ds = MultiConditionAnnotatedDataset(
            adata,
            self.condition_keys_,
            self.model.condition_encoders if condition_encoders is None else condition_encoders,
            self.model.conditions_combined_encoder if conditions_combined_encoder is None else conditions_combined_encoder,
        )
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate
        )

        device = next(self.model.parameters()).device
        recon_loss = 0.
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, recon_loss_batch, _, _ = self.model(**batch)
                recon_loss += recon_loss_batch * batch["x"].shape[0]

        return (recon_loss / len(ds)).item()

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new embedding in dictionary
            elif "embedding" in key:
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[0] - load_ten.size()[0]
                fixed_ten = torch.cat([load_ten, new_ten[-dim_diff:, ...]], dim=0)
                load_state_dict[key] = fixed_ten
            else:
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        self.model.load_state_dict(load_state_dict)
