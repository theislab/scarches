import numpy as np
from typing import Union, List, Dict
import torch
import anndata

from scarches.models import SCVI, SCANVI, TOTALVI
from scarches.trainers import scVITrainer, scANVITrainer, totalTrainer

from scvi.data import transfer_anndata_setup
from scvi.core.models._utils import _initialize_model
from scvi.core.models.archesmixin import _set_params_online_update

def _transfer_model(model, adata):
    adata = adata.copy()

    attr_dict = model._get_user_attributes()
    attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
    scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")
    transfer_anndata_setup(scvi_setup_dict, adata, extend_categories=True)

    adata.uns["_scvi"]["summary_stats"]["n_labels"] = scvi_setup_dict[
        "summary_stats"
    ]["n_labels"]

    new_model = _initialize_model(model.__class__, adata, attr_dict, use_cuda=True)

    for attr, val in attr_dict.items():
        setattr(new_model, attr, val)

    model.model.cuda()
    new_model.model.cuda()
    new_state_dict = model.model.state_dict()

    load_state_dict = model.model.state_dict().copy()
    new_state_dict = new_model.model.state_dict()
    for key, load_ten in load_state_dict.items():
        new_ten = new_state_dict[key]
        if new_ten.size() == load_ten.size():
            continue
        else:
            dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
            fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
            load_state_dict[key] = fixed_ten
    new_model.model.load_state_dict(load_state_dict)
    new_model.model.eval()

    new_model.is_trained_ = False

    return new_model, adata

def scvi_operate(
        model: Union[SCVI, SCANVI, TOTALVI],
        data: anndata.AnnData,
        labels_per_class: int = 0,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        early_stopping: bool = False,
        freeze: bool = True,
        freeze_expression: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_dropout: bool = False,
        **kwargs
) -> [Union[SCVI, SCANVI, TOTALVI], Union[scVITrainer, scANVITrainer, totalTrainer], anndata.AnnData]:
    """Transfer Learning function for new data. Uses old trained Network and expands it for new conditions.
       Parameters
       ----------
       model: VAE_M, SCANVI_M, TOTALVI_M
            A Scvi/Scanvi/Totalvi model object
       data: AnnData
            AnnData object.
       labels_per_class: Integer
            Number of labelled Samples used for Retraining
       n_epochs: Integer
            Number of epochs for training the network on query data.
       freeze: Boolean
            If 'True' freezes every part of the network except predefined layers dependent of model version.
       freeze_expression: Boolean
            If 'True' freeze every weight except the condition weights.
       freeze_batchnorm_encoder: Boolean
            If 'True' freeze Batchnorm in the encoder for Transfer Learning.
       freeze_batchnorm_decoder: Boolean
            If 'True' freeze Batchnorm in the decoder for Transfer Learning.
       freeze_dropout: Boolean
            If 'True' remove Dropout for Transfer Learning.

       Returns
       -------
       new_model: SCVI, SCANVI, TOTALVI
            Newly network that got trained on query data.
       op_trainer: scVITrainer, scANVITrainer, totalTrainer
            Trainer for the newly network.
       data: AnnData
            AnnData object with updated batch labels.
    """

    if not early_stopping:
        early_stopping_kwargs = None
    else:
        if type(model) is not SCANVI:
            early_stopping_kwargs = {
                "early_stopping_metric": "elbo",
                "save_best_state_metric": "elbo",
                "patience": 15,
                "threshold": 0,
                "reduce_lr_on_plateau": True,
                "lr_patience": 8,
                "lr_factor": 0.1,
            }
        else:
            early_stopping_kwargs = {
                "early_stopping_metric": "accuracy",
                "save_best_state_metric": "accuracy",
                "on": "full_dataset",
                "patience": 15,
                "threshold": 0.001,
                "reduce_lr_on_plateau": True,
                "lr_patience": 8,
                "lr_factor": 0.1,
            }

    new_model, adata = _transfer_model(model, data)
    if freeze:
        _set_params_online_update(
            new_model.model,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
        )

    # Retrain Networks
    if type(new_model) is SCVI:
        op_trainer = scVITrainer(
            new_model,
            train_size=0.9,
            use_cuda=True,
            frequency=1,
            early_stopping_kwargs=early_stopping_kwargs,
            **kwargs
        )
    if type(new_model) is SCANVI:
        op_trainer = scANVITrainer(
            new_model,
            n_labelled_samples_per_class=labels_per_class,
            train_size=0.9,
            use_cuda=True,
            frequency=1,
            early_stopping_kwargs=early_stopping_kwargs,
            **kwargs
        )
    if type(new_model) is TOTALVI:
        op_trainer = totalTrainer(
            new_model,
            train_size=0.9,
            use_cuda=True,
            frequency=1,
            early_stopping_kwargs=early_stopping_kwargs,
            **kwargs
        )

    op_trainer.train(n_epochs=n_epochs, lr=learning_rate)

    return new_model, op_trainer, adata
