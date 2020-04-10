from __future__ import absolute_import, division, print_function

import scanpy as sc
from keras.utils import to_categorical
from kipoi.data import Dataset
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from scnet.utils import remove_sparsity, normalize


class KipoiDatasetWrapper(Dataset):
    """Example re-implementation of kipoiseq.dataloaders.SeqIntervalDl

    Args:
        filename: h5ad data file path (Read count or normalized expression)
        batch_key: column of batch (studies) labels
        count: is True if data is count matrix

    """

    def __init__(self, filename, batch_key, count=True):
        self.adata = sc.read(filename)
        self.batch_labels = self.adata.obs[batch_key].values
        self.n_batches = len(self.adata.obs[batch_key].unique().tolist())
        self.count = count

        self.adata = remove_sparsity(self.adata)
        self.batch_encoder = LabelEncoder().fit(self.batch_labels)

        if self.count:
            if self.adata.raw is not None and sparse.issparse(self.adata.raw.X):
                self.adata.raw.X = self.adata.raw.X.A

            self.adata = normalize(self.adata, filter_min_counts=False,
                                   size_factors=True, normalize_input=False,
                                   logtrans_input=True, n_top_genes=5000)
            self.size_factors = self.adata.obs['size_factors']

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        genes = self.adata.X[idx, :]
        batch = self.batch_labels[idx]

        batch_encoded = self.batch_encoder.transform(batch)
        batch_onehot = to_categorical(batch_encoded, num_classes=self.n_batches)

        size_factor = self.size_factors[idx]

        if self.count:
            predicted_genes = self.adata.raw.X[idx]
        else:
            predicted_genes = self.adata.X[idx]

        return {
            "inputs": {
                "genes": genes,
                "study": batch_onehot,
                "size_factors": size_factor
            },
            "targets": {
                "predicted_genes": predicted_genes,
            },
        }
