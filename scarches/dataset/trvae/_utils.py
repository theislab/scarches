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

    labels = np.zeros(adata.shape[0])
    if not set(unique_conditions).issubset(set(encoder.keys())):
        print("Warning: Labels in adata is not a subset of label-encoder!")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), encoder
