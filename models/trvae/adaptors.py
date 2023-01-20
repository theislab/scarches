import os
import numpy as np
import torch
import pickle
from typing import Optional, Union
from copy import deepcopy
from .trvae_model import TRVAE

class Adaptor:
    """Adaptor class for trVAE.

       Allows to save and load trainded conditional weights for trVAE models.

       Parameters
       ----------
       trvae_model
            A TRVAE class object with a trainded model or a path to saved Adaptor object.
       condition
            Condition name to save in the adaptor.
    """
    model_type = 'trVAE'

    def __init__(
        self,
        trvae_model: Union[str, TRVAE],
        condition: Optional[str] = None
    ):
        if isinstance(trvae_model, str):
            cond_params_path = os.path.join(trvae_model, "cond_params.pt")
            adapt_params_path = os.path.join(trvae_model, "adapt_params.pkl")

            self.cond_params = torch.load(cond_params_path)

            with open(adapt_params_path, "rb") as handle:
                self._adapt_params = pickle.load(handle)

            self.condition = self._adapt_params['condition']
        else:
            self.cond_params = {}
            self.condition = condition

            cond_idx = trvae_model.conditions_.index(self.condition)
            for name, p in trvae_model.model.state_dict().items():
                if 'cond_L.weight' in name or 'theta' in name:
                    self.cond_params[name] = p[:, cond_idx].unsqueeze(-1)

            self._adapt_params = {}

            self._adapt_params['condition'] = self.condition

            self._adapt_params['model_params'] = {}

            self._adapt_params['model_params']['varnames'] = trvae_model.adata.var_names.tolist()

            self._adapt_params['model_params']['hidden_layer_sizes'] = trvae_model.hidden_layer_sizes_
            self._adapt_params['model_params']['latent_dim'] = trvae_model.latent_dim_
            self._adapt_params['model_params']['recon_loss'] = trvae_model.recon_loss_

    def _validate_params(self, varnames, init_params):
        params = self._adapt_params['model_params'].copy()

        adaptor_varnames = np.array(params.pop('varnames'), dtype=str)
        if not np.array_equal(adaptor_varnames, varnames.astype(str)):
            logger.warning(
                "var_names for adata in the model does not match var_names of "
                "adata used to train the model of the adaptor. For valid results, the vars "
                "need to be the same and in the same order as the adata used to train the model."
            )

        for k in params:
            if init_params[k] != params[k]:
                raise ValueError(f'Parameter {k} in the adaptor isn\'t equal to {k} of the model.')

    def save(
        self,
        dir_path: str,
        overwrite: Optional[bool] = False
    ):
        """Save the state of the adaptor.

           Parameters
           ----------
           dir_path
                Path to a directory.
           overwrite
                Overwrite existing data or not. If `False` and directory
                already exists at `dir_path`, error will be raised.
        """
        cond_params_path = os.path.join(dir_path, "cond_params.pt")
        adapt_params_path = os.path.join(dir_path, "adapt_params.pkl")

        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                f"{dir_path} already exists. Please provide an unexisting directory for saving."
            )

        torch.save(self.cond_params, cond_params_path)

        with open(adapt_params_path, "wb") as f:
            pickle.dump(self._adapt_params, f)


def attach_adaptors(
    trvae_model: TRVAE,
    adaptors: list,
    only_new: bool = False
):
    """Attach the conditional weights from the adaptors to a trVAE model.

       Attaches the conditional weights saved in the adaptors to a model,
       expanding it to all conditions present in the adaptors.

       Parameters
       ----------
       trvae_model
            A TRVAE class object. The object should have the same architecture
            as the model which was used to save the conditional weights to the adaptors.
       adaptors
            List of adaptors to attach.
       only_new
            Attach only condtional weights for new conditions.
            Do not overwrite conditional weights for the conditions
            which are already in the model (in `trvae_model.conditions_`).
    """
    attr_dict = trvae_model._get_public_attributes()
    init_params = deepcopy(TRVAE._get_init_params_from_dict(attr_dict))

    adpt_conditions = []
    cond_params = {}

    for adaptor in adaptors:
        if isinstance(adaptor, str):
            adaptor = Adaptor(adaptor)
        adaptor._validate_params(trvae_model.adata.var_names, init_params)
        adpt_conditions.append(adaptor.condition)
        for k, p in adaptor.cond_params.items():
            if k not in cond_params:
                cond_params[k] = p.clone()
            else:
                cond_params[k] = torch.cat([cond_params[k], p], dim=-1)

    inds_exist, inds_old, inds_new = [], [], []

    conditions = init_params['conditions']
    for i, c in enumerate(adpt_conditions):
        if c not in conditions:
            inds_new.append(i)
        else:
            inds_exist.append(i)
            inds_old.append(conditions.index(c))

    init_params['conditions'] += [adpt_conditions[i] for i in inds_new]

    new_model = TRVAE(trvae_model.adata, **init_params)
    state_dict = trvae_model.model.state_dict().copy()

    for k, ten in cond_params.items():
        new_ten = state_dict[k]
        if not only_new and len(inds_exist) > 0:
            new_ten[:, inds_old] = ten[:, inds_exist]
        if len(inds_new) > 0:
            state_dict[k] = torch.cat([new_ten, ten[:, inds_new]], dim=-1)
    new_model.model.load_state_dict(state_dict)

    return new_model
