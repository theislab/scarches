from collections import Counter

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
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None
                 ):

        self.X_norm = None

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        if sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        self.data = torch.tensor(X)

        size_factors = adata.X.sum(1)
        if len(size_factors.shape) < 2:
            size_factors = np.expand_dims(size_factors, axis=1)

        self.size_factors = torch.tensor(size_factors)

        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        self.labeled_vector = torch.tensor(labeled_array)

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

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
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)
