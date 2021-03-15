import os
import numpy as np
import torch
import pickle
from typing import Optional, Union
from .trvae_model import TRVAE

class Adaptor:
    """Adaptor class for trVAE.

       Allows to save and load trainded conditional weights for trVAE models.

       Parameters
       ----------
       trvae_model
            A TRVAE class object with a trainded model or a path to saved Adaptor object.
    """
    model_type = 'trVAE'

    def __init__(
        self,
        trvae_model: Union[str, TRVAE]
    ):
        if isinstance(trvae_model, str):
            cond_params_path = os.path.join(trvae_model, "cond_params.pt")
            cond_names_path = os.path.join(trvae_model, "cond_names.csv")
            architec_params_path = os.path.join(trvae_model, "architec_params.pkl")

            self.cond_params = torch.load(cond_params_path)

            self.conditions = np.genfromtxt(cond_names_path, delimiter=",", dtype=str)
            self.conditions = self.conditions.tolist()

            with open(architec_params_path, "rb") as handle:
                self.architec_params = pickle.load(handle)
        else:
            self.cond_params = {}
            self.conditions = trvae_model.conditions_.copy()

            for name, p in trvae_model.model.state_dict().items():
                if 'cond_L.weight' in name or 'theta' in name:
                    self.cond_params[name] = p.clone()

            self.architec_params = {}

            self.architec_params['varnames'] = trvae_model.adata.var_names.tolist()

            self.architec_params['hidden_layer_sizes'] = trvae_model.hidden_layer_sizes_
            self.architec_params['latent_dim'] = trvae_model.latent_dim_
            self.architec_params['recon_loss'] = trvae_model.recon_loss_

    def _validate_params(self, varnames, init_params):
        params = self.architec_params.copy()

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


    def attach(
        self,
        trvae_model: TRVAE,
        only_new: Optional[bool] = False
    ):
        """Attach the conditional weights from the adaptor to a trVAE model.

           Attaches the conditional weights saved in the adaptor to a model,
           expanding it to all conditions present in the adaptor.

           Parameters
           ----------
           trvae_model
                A TRVAE class object. The object should have the same architecture
                as the model which was used to save the conditional weights to the adaptor.
           only_new
                Attach only condtional weights for new conditions.
                Do not overwrite conditional weights for the conditions
                which are already in the model (in `trvae_model.conditions_`).
        """
        attr_dict = trvae_model._get_public_attributes()
        init_params = TRVAE._get_init_params_from_dict(attr_dict)

        self._validate_params(trvae_model.adata.var_names, init_params)

        inds_exist, inds_old, inds_new = [], [], []

        conditions = init_params['conditions']
        for i, c in enumerate(self.conditions):
            if c not in conditions:
                inds_new.append(i)
            else:
                inds_exist.append(i)
                inds_old.append(conditions.index(c))

        init_params['conditions'] += [self.conditions[i] for i in inds_new]

        new_model = TRVAE(trvae_model.adata, **init_params)
        state_dict = trvae_model.model.state_dict().copy()

        for k, ten in self.cond_params.items():
            new_ten = state_dict[k]
            if not only_new and len(inds_exist) > 0:
                new_ten[:, inds_old] = ten[:, inds_exist]
            if len(inds_new) > 0:
                state_dict[k] = torch.cat([new_ten, ten[:, inds_new]], dim=-1)
        new_model.model.load_state_dict(state_dict)

        return new_model

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
        cond_names_path = os.path.join(dir_path, "cond_names.csv")
        architec_params_path = os.path.join(dir_path, "architec_params.pkl")

        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)

        torch.save(self.cond_params, cond_params_path)
        np.savetxt(cond_names_path, self.conditions, fmt="%s")

        with open(architec_params_path, "wb") as f:
            pickle.dump(self.architec_params, f)
