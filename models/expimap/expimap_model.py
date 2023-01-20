import inspect
import os
import torch
import pickle
import numpy as np
import pandas as pd

from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union

from .expimap import expiMap
from ...trainers import expiMapTrainer
from ..base._utils import _validate_var_names
from ..base._base import BaseMixin, SurgeryMixin, CVAELatentsMixin


class EXPIMAP(BaseMixin, SurgeryMixin, CVAELatentsMixin):
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
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse' or 'nb'.
       use_l_encoder: Boolean
            If True and `decoder_last_layer`='softmax', libary size encoder is used.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       mask: Array or List
            if not None, an array of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
       mask_key: String
            A key in `adata.varm` for the mask if the mask is not provided.
       decoder_last_layer: String or None
            The last layer of the decoder. Must be 'softmax' (default for 'nb' loss), identity(default for 'mse' loss),
            'softplus', 'exp' or 'relu'.
       soft_mask: Boolean
            Use soft mask option. If True, the model will enforce mask with L1 regularization
            instead of multipling weight of the linear decoder by the binary mask.
       n_ext: Integer
            Number of unconstarined extension terms.
            Used for query mapping.
       n_ext_m: Integer
            Number of constrained extension terms.
            Used for query mapping.
       use_hsic: Boolean
            If True, add HSIC regularization for unconstarined extension terms.
            Used for query mapping.
       hsic_one_vs_all: Boolean
            If True, calculates the sum of HSIC losses for each unconstarined term vs the other terms.
            If False, calculates HSIC for all unconstarined terms vs the other terms.
            Used for query mapping.
       ext_mask: Array or List
            Mask (similar to the mask argument) for unconstarined extension terms.
            Used for query mapping.
       soft_ext_mask: Boolean
            Use the soft mask mode for training with the constarined extension terms.
            Used for query mapping.
    """
    def __init__(
        self,
        adata: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        hidden_layer_sizes: list = [256, 256],
        dr_rate: float = 0.05,
        recon_loss: str = 'nb',
        use_l_encoder: bool = False,
        use_bn: bool = False,
        use_ln: bool = True,
        mask: Optional[Union[np.ndarray, list]] = None,
        mask_key: str = 'I',
        decoder_last_layer: Optional[str] = None,
        soft_mask: bool = False,
        n_ext: int = 0,
        n_ext_m: int = 0,
        use_hsic: bool = False,
        hsic_one_vs_all: bool = False,
        ext_mask: Optional[Union[np.ndarray, list]] = None,
        soft_ext_mask: bool = False
    ):
        self.adata = adata

        if mask is None and mask_key not in self.adata.varm:
            raise ValueError('Please provide mask.')

        self.condition_key_ = condition_key

        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.dr_rate_ = dr_rate
        self.recon_loss_ = recon_loss
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln

        self.input_dim_ = adata.n_vars

        self.use_l_encoder_ = use_l_encoder
        self.decoder_last_layer_ = decoder_last_layer

        if mask is None:
            mask = adata.varm[mask_key].T

        self.mask_ = mask if isinstance(mask, list) else mask.tolist()
        mask = torch.tensor(mask).float()
        self.latent_dim_ = len(self.mask_)

        self.ext_mask_ = None
        if ext_mask is not None:
            self.ext_mask_ = ext_mask if isinstance(ext_mask, list) else ext_mask.tolist()
            ext_mask = torch.tensor(ext_mask).float()

        self.n_ext_ = n_ext
        self.n_ext_m_ = n_ext_m

        self.soft_mask_ = soft_mask
        self.soft_ext_mask_ = soft_ext_mask

        self.use_hsic_ = use_hsic and n_ext > 0
        self.hsic_one_vs_all_ = hsic_one_vs_all

        self.model = expiMap(
            self.input_dim_,
            self.latent_dim_,
            mask,
            self.conditions_,
            self.hidden_layer_sizes_,
            self.dr_rate_,
            self.recon_loss_,
            self.use_l_encoder_,
            self.use_bn_,
            self.use_ln_,
            self.decoder_last_layer_,
            self.soft_mask_,
            self.n_ext_,
            self.n_ext_m_,
            self.use_hsic_,
            self.hsic_one_vs_all_,
            ext_mask,
            self.soft_ext_mask_
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
        **kwargs
    ):
        """Train the model.

           Parameters
           ----------
           n_epochs: Integer
                Number of epochs for training the model.
           lr: Float
                Learning rate for training the model.
           eps: Float
                torch.optim.Adam eps parameter
           alpha_kl: Float
                Multiplies the KL divergence part of the loss.
           alpha: Float
                Group Lasso regularization coefficient
           omega: Tensor or None
                If not 'None', vector of coefficients for each group
           alpha_l1: Float
                L1 regularization coefficient for the soft mask of reference (old) and new constrained terms.
                Specifies the strength for deactivating the genes which are not in the corresponding annotations \ groups
                in the mask.
           alpha_l1_epoch_anneal: Integer
                If not 'None', the alpha_l1 scaling factor will be annealed from 0 to 1 every 'alpha_l1_anneal_each' epochs
                until the input integer is reached.
           alpha_l1_anneal_each: Integer
                Anneal alpha_l1 every alpha_l1_anneal_each'th epoch, i.e. for 5 (default)
                do annealing every 5th epoch.
           gamma_ext: Float
                L1 regularization coefficient for the new unconstrained terms. Specifies the strength of
                sparcity enforcement.
           gamma_epoch_anneal: Integer
                If not 'None', the gamma_ext scaling factor will be annealed from 0 to 1 every 'gamma_anneal_each' epochs
                until the input integer is reached.
           gamma_anneal_each: Integer
                Anneal gamma_ext every gamma_anneal_each'th epoch, i.e. for 5 (default)
                do annealing every 5th epoch.
           beta: Float
                HSIC regularization coefficient for the unconstrained terms.
                Multiplies the HSIC loss terms if not 'None'.
           kwargs
                kwargs for the expiMap trainer.
        """
        self.trainer = expiMapTrainer(
            self.model,
            self.adata,
            alpha=alpha,
            omega=omega,
            condition_key=self.condition_key_,
            **kwargs
        )
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    def nonzero_terms(self):
        """Return indices of active terms.
           Active terms are the terms which were not deactivated by the group lasso regularization.
        """
        return self.model.decoder.nonzero_terms()

    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        only_active: bool = False,
        mean: bool = False,
        mean_var: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder
           and return z for each sample in data.

           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           only_active
                Return only the latent variables which correspond to active terms, i.e terms that
                were not deactivated by the group lasso regularization.
           mean
                return mean instead of random sample from the latent space
           mean_var
                return mean and variance instead of random sample from the latent space
                if `mean=False`.

           Returns
           -------
                Returns array containing latent space encoding of 'x'.
        """
        result = super().get_latent(x, c, mean, mean_var)

        if not only_active:
            return result
        else:
            active_idx = self.nonzero_terms()
            if isinstance(result, tuple):
                result = tuple(r[:, active_idx] for r in result)
            else:
                result = result[:, active_idx]
            return result

    def update_terms(self, terms: Union[str, list]='terms', adata=None):
        """Add extension terms' names to the terms.
        """
        if isinstance(terms, str):
            adata = self.adata if adata is None else adata
            key = terms
            terms = list(adata.uns[terms])
        else:
            adata = None
            key = None
            terms = list(terms)

        lat_mask_dim = self.latent_dim_ + self.n_ext_m_
        if len(terms) != self.latent_dim_ and len(terms) != lat_mask_dim + self.n_ext_:
            raise ValueError('The list of terms should have the same length as the mask.')

        if len(terms) == self.latent_dim_:
            if self.n_ext_m_ > 0:
                terms += ['constrained_' + str(i) for i in range(self.n_ext_m_)]
            if self.n_ext_ > 0:
                terms += ['unconstrained_' + str(i) for i in range(self.n_ext_)]

        if adata is not None:
            adata.uns[key] = terms
        else:
            return terms

    def term_genes(self, term: Union[str, int], terms: Union[str, list]='terms'):
        """Return the dataframe with genes belonging to the term after training sorted by absolute weights in the decoder.
        """
        if isinstance(terms, str):
            terms = list(self.adata.uns[terms])
        else:
            terms = list(terms)

        if len(terms) == self.latent_dim_:
            if self.n_ext_m_ > 0:
                terms += ['constrained_' + str(i) for i in range(self.n_ext_m_)]
            if self.n_ext_ > 0:
                terms += ['unconstrained_' + str(i) for i in range(self.n_ext_)]

        lat_mask_dim = self.latent_dim_ + self.n_ext_m_

        if len(terms) != self.latent_dim_ and len(terms) != lat_mask_dim + self.n_ext_:
            raise ValueError('The list of terms should have the same length as the mask.')

        term = terms.index(term) if isinstance(term, str) else term

        if term < self.latent_dim_:
            weights = self.model.decoder.L0.expr_L.weight[:, term].data.cpu().numpy()
            mask_idx = self.mask_[term]
        elif term >= lat_mask_dim:
            term -= lat_mask_dim
            weights = self.model.decoder.L0.ext_L.weight[:, term].data.cpu().numpy()
            mask_idx = None
        else:
            term -= self.latent_dim_
            weights = self.model.decoder.L0.ext_L_m.weight[:, term].data.cpu().numpy()
            mask_idx = self.ext_mask_[term]

        abs_weights = np.abs(weights)
        srt_idx = np.argsort(abs_weights)[::-1][:(abs_weights > 0).sum()]

        result = pd.DataFrame()
        result['genes'] = self.adata.var_names[srt_idx].tolist()
        result['weights'] = weights[srt_idx]
        result['in_mask'] = False

        if mask_idx is not None:
            in_mask = np.isin(srt_idx, np.where(mask_idx)[0])
            result['in_mask'][in_mask] = True

        return result

    def mask_genes(self, terms: Union[str, list]='terms'):
        """Return lists of genes belonging to the terms in the mask.
        """
        if isinstance(terms, str):
            terms = list(self.adata.uns[terms])
        else:
            terms = list(terms)

        I = np.array(self.mask_)

        if self.n_ext_m_ > 0:
            I = np.concatenate((I, self.ext_mask_))

            if len(terms) == self.latent_dim_:
                terms += ['constrained_' + str(i) for i in range(self.n_ext_m_)]
            elif len(terms) == self.latent_dim_ + self.n_ext_m_ + self.n_ext_:
                terms = terms[:(self.latent_dim_ + self.n_ext_m_)]
            else:
                raise ValueError('The list of terms should have the same length as the mask.')

        I = I.astype(bool)

        return {term: self.adata.var_names[I[i]].tolist() for i, term in enumerate(terms)}

    def latent_directions(self, method="sum", get_confidence=False,
                          adata=None, key_added='directions'):
        """Get directions of upregulation for each latent dimension.
           Multipling this by raw latent scores ensures positive latent scores correspond to upregulation.

           Parameters
           ----------
           method: String
                Method of calculation, it should be 'sum' or 'counts'.
           get_confidence: Boolean
                Only for method='counts'. If 'True', also calculate confidence
                of the directions.
           adata: AnnData
                An AnnData object to store dimensions. If 'None', self.adata is used.
           key_added: String
                key of adata.uns where to put the dimensions.
        """
        if adata is None:
            adata = self.adata

        terms_weights = self.model.decoder.L0.expr_L.weight.data
        if self.n_ext_m_ > 0:
            terms_weights = torch.cat([terms_weights, self.model.decoder.L0.ext_L_m.weight.data], dim=1)
        if self.n_ext_ > 0:
            terms_weights = torch.cat([terms_weights, self.model.decoder.L0.ext_L.weight.data], dim=1)

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

        adata.uns[key_added] = signs
        if get_confidence and confidence is not None:
            adata.uns[key_added + '_confindence'] = confidence

    def latent_enrich(
        self,
        groups,
        comparison='rest',
        n_sample=5000,
        use_directions=False,
        directions_key='directions',
        select_terms=None,
        adata=None,
        exact=True,
        key_added='bf_scores'
    ):
        """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
           for each term in one group (z_1) is bigger than in the other group (z_2).

           Puts results to `adata.uns[key_added]`. Results are a dictionary with
           `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.

           Parameters
           ----------
           groups: String or Dict
                A string with the key in `adata.obs` to look for categories or a dictionary
                with categories as keys and lists of cell names as values.
           comparison: String
                The category name to compare against. If 'rest', then compares each category against all others.
           n_sample: Integer
                Number of random samples to draw for each category.
           use_directions: Boolean
                If 'True', multiplies the latent scores by directions in `adata`.
           directions_key: String
                The key in `adata.uns` for directions.
           select_terms: Array
                If not 'None', then an index of terms to select for the test. Only does the test
                for these terms.
           adata: AnnData
                An AnnData object to use. If 'None', uses `self.adata`.
           exact: Boolean
                Use exact probabilities for comparisons.
           key_added: String
                key of adata.uns where to put the results of the test.
        """
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

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        if comparison != "rest" and not set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            choice_1 = np.random.choice(cat_mask.sum(), n_sample)
            choice_2 = np.random.choice(others_mask.sum(), n_sample)

            adata_cat = adata[cat_mask][choice_1]
            adata_others = adata[others_mask][choice_2]

            if use_directions:
                directions = adata.uns[directions_key]
            else:
                directions = None

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

                if select_terms is not None:
                    means0 = means0[:, select_terms]
                    means1 = means1[:, select_terms]
                    vars0 = vars0[:, select_terms]
                    vars1 = vars1[:, select_terms]

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

        adata.uns[key_added] = scores

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'TRVAE'],
        freeze: bool = True,
        freeze_expression: bool = True,
        unfreeze_ext: bool = True,
        remove_dropout: bool = True,
        new_n_ext: Optional[int] = None,
        new_n_ext_m: Optional[int] = None,
        new_ext_mask: Optional[Union[np.ndarray, list]] = None,
        new_soft_ext_mask: bool = False,
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
           unfreeze_ext: Boolean
                If 'True' do not freeze weights for new constrained and unconstrained extension terms.
           new_n_ext: Integer
                Number of new unconstarined extension terms to add to the reference model.
                Used for query mapping.
           new_n_ext_m: Integer
                Number of new constrained extension terms to add to the reference model.
                Used for query mapping.
           new_ext_mask: Array or List
                Mask (similar to the mask argument) for new unconstarined extension terms.
           new_soft_ext_mask: Boolean
                Use the soft mask mode for training with the constarined extension terms.
           kwargs
                kwargs for the initialization of the EXPIMAP class for the query model.

           Returns
           -------
           new_model
                New (query) model to train on query data.
        """
        params = {}
        params['adata'] = adata
        params['reference_model'] = reference_model
        params['freeze'] = freeze
        params['freeze_expression'] = freeze_expression
        params['remove_dropout'] = remove_dropout

        if new_n_ext is not None:
            params['n_ext'] = new_n_ext
        if new_n_ext_m is not None:
            params['n_ext_m'] = new_n_ext_m
            if new_ext_mask is None:
                raise ValueError('Provide new ext_mask')
            params['ext_mask'] = new_ext_mask
            params['soft_ext_mask'] = new_soft_ext_mask

        params.update(kwargs)

        new_model = super().load_query_data(**params)

        if freeze and unfreeze_ext:
            for name, p in new_model.model.named_parameters():
                if 'ext_L.weight' in name or 'ext_L_m.weight' in name:
                    p.requires_grad = True
                if 'expand_mean_encoder' in name or 'expand_var_encoder' in name:
                    p.requires_grad = True

        return new_model

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_key': dct['condition_key_'],
            'conditions': dct['conditions_'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'dr_rate': dct['dr_rate_'],
            'recon_loss': dct['recon_loss_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
            'mask': dct['mask_'],
            'decoder_last_layer': dct['decoder_last_layer_'] if 'decoder_last_layer_' in dct else "softmax",
            'use_l_encoder': dct['use_l_encoder_'] if 'use_l_encoder_' in dct else False,
            'n_ext': dct['n_ext_'] if 'n_ext_' in dct else 0,
            'n_ext_m': dct['n_ext_m_'] if 'n_ext_m_' in dct else 0,
            'soft_mask': dct['soft_mask_'] if 'soft_mask_' in dct else False,
            'soft_ext_mask': dct['soft_ext_mask_'] if 'soft_ext_mask_' in dct else False,
            'hsic_one_vs_all': dct['hsic_one_vs_all_'] if 'hsic_one_vs_all_' in dct else False,
            'use_hsic': dct['use_hsic_'] if 'use_hsic_' in dct else False,
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
