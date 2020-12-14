import numpy as np


def label_encoder(adata, encoder=None, condition_key='condition'):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict or None
            dictionary of encoded labels. if `None`, will create one.
       condition_key: String
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    if encoder is None:
        encoder = {k: v for k, v in zip(sorted(unique_conditions), np.arange(len(unique_conditions)))}

    assert set(unique_conditions).issubset(set(encoder.keys()))
    labels = np.zeros(adata.shape[0])
    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), encoder
