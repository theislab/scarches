import numpy as np


def label_encoder(adata, encoder, condition_key=None):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
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
    labels = np.zeros(adata.shape[0]) #result in a vector of (15681,)

    if not set(unique_conditions).issubset(set(encoder.keys())):
        print(f"Warning: Labels in adata.obs[{condition_key}] is not a subset of label-encoder!")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items(): 
        labels[adata.obs[condition_key] == condition] = label # if source_conditions is True, then all the values (zeros) of the labels vector
                                                              # will take the value of label which the value of the key in encoder.items()
                                                       # and this is how you lable each cell to the corresponding condition (study)
    return labels
