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
       condition_encoder: Dict or None
            dictionary of encoded conditions. if `None`, will create one.
       cell_type_key: String
            column name of celltypes in `adata.obs` data frame.
       cell_type_encoder: Dict or None
            dictionary of encoded celltypes. if `None`, will create one.
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_key=None,
                 cell_type_encoder=None,
                 ):

        self.X_norm = None

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_key = cell_type_key
        self.cell_type_encoder = cell_type_encoder

        if sparse.issparse(adata.X):
            adata = remove_sparsity(adata)
        self.data = torch.tensor(adata.X)

        self.size_factors = torch.tensor(adata.obs['trvae_size_factors'])
        self.labeled_vector = torch.tensor(adata.obs['trvae_labeled'])

        # Create Condition Encoder
        if self.condition_key is not None:
            self.conditions, self.condition_encoder = label_encoder(adata,
                                                                    encoder=self.condition_encoder,
                                                                    condition_key=condition_key)
            self.conditions = torch.tensor(np.array(self.conditions).reshape(-1, ), dtype=torch.long)

        # Create Cell Type Encoder
        if self.cell_type_key is not None:
            self.cell_types, self.cell_type_encoder = label_encoder(adata,
                                                                    encoder=self.cell_type_encoder,
                                                                    condition_key=cell_type_key)
            self.cell_types = torch.tensor(np.array(self.cell_types).reshape(-1, ), dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        outputs["x"] = self.data[index, :]
        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_key:
            outputs["celltype"] = self.cell_types[index]

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
