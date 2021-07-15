import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse

from .data_handling import remove_sparsity
from ._utils import label_encoder


class AnnotatedDataset(Dataset):
    """Dataset handler for TRVAE model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
       condition_weights: Dict
            Weight samples' contribution to loss based on their condition weight.
            Dictionary with K: condition, V: weight.
            If None perform no sample weighting (e.g. all weights are set to 1).
    """

    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 condition_weights=None,
                 ):

        self.X_norm = None

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder
        self.condition_weights = condition_weights

        if sparse.issparse(adata.X):
            adata = remove_sparsity(adata)
        self.data = torch.tensor(adata.X)

        self.size_factors = torch.tensor(adata.obs['trvae_size_factors'])
        self.labeled_vector = torch.tensor(adata.obs['trvae_labeled'])

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

        # Encode sample weights based on condition weights
        if self.condition_key is not None and self.condition_weights is not None:
            weights = [self.condition_weights[condition] for condition in adata.obs[condition_key]]
            # TODO add warning if some conditions are not in weights dict
        else:
            weights = np.ones(adata.shape[0])
        self.sample_weights = torch.tensor(weights)

        # Encode cell type strings to integer
        if self.cell_type_keys is not None:
            self.cell_types = list()
            for cell_type_key in cell_type_keys:
                level_cell_types = label_encoder(
                    adata,
                    encoder=self.cell_type_encoder,
                    condition_key=cell_type_key,
                )
                self.cell_types.append(level_cell_types)

            self.cell_types = np.stack(self.cell_types).T
            self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        outputs["x"] = self.data[index, :]
        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_keys:
            outputs["celltypes"] = self.cell_types[index, :]

        outputs["sampleweight"] = self.sample_weights[index]

        return outputs

    def __len__(self):
        return self.data.size(0)

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1 / len(conditions)
        weights_per_condition = list()
        for i in range(len(self.conditions)):
            samples_per_condition = np.count_nonzero(conditions == i)
            if samples_per_condition == 0:
                weights_per_condition.append(0)
            else:
                weights_per_condition.append((1 / samples_per_condition) * condition_coeff)
        strat_weights = np.copy(conditions)
        for i in range(len(conditions)):
            strat_weights = np.where(strat_weights == i, weights_per_condition[i], strat_weights)

        return strat_weights.astype(float)
