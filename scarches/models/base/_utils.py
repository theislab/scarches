import io
import logging
import pickle

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack

logger = logging.getLogger(__name__)


def _validate_var_names(adata, source_var_names):
    #Warning for gene percentage
    user_var_names = adata.var_names
    try:
        percentage = (len(user_var_names.intersection(source_var_names)) / len(user_var_names)) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(f"WARNING: Query shares {percentage}% of its genes with the reference."
                            "This may lead to inaccuracy in the results.")
    except Exception:
            logger.warning("WARNING: Something is wrong with the reference genes.")

    user_var_names = user_var_names.astype(str)
    new_adata = adata

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in source_var_names:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print("Query data is missing expression data of ",
              len(ref_genes_not_in_query),
              " genes which were contained in the reference dataset.")
        print("The missing information will be filled with zeroes.")
       
        filling_X = np.zeros((len(adata), len(ref_genes_not_in_query)))
        if isinstance(adata.X, csr_matrix): 
            filling_X = csr_matrix(filling_X) # support csr sparse matrix
            new_target_X = hstack((adata.X, filling_X))
        else:
            new_target_X = np.concatenate((adata.X, filling_X), axis=1)
        new_target_vars = adata.var_names.tolist() + ref_genes_not_in_query
        new_adata = AnnData(new_target_X, dtype="float32")
        new_adata.var_names = new_target_vars
        new_adata.obs = adata.obs.copy()

    if len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)) > 0:
        print(
            "Query data contains expression data of ",
            len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)),
            " genes that were not contained in the reference dataset. This information "
            "will be removed from the query data object for further processing.")

        # remove unseen gene information and order anndata
        new_adata = new_adata[:, source_var_names].copy()

    print(new_adata)

    return new_adata

def get_minified_adata_scrna(
    adata: AnnData,
) -> AnnData:


    """This function is adapted from scvi-tools
    https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.utils.get_minified_adata_scrna.html
    
    Returns a minified adata.
    Parameters
    ----------
    adata
        Original adata, of which we to create a minified version.
    """
    all_zeros = csr_matrix(adata.X.shape)
    layers = {layer: all_zeros.copy() for layer in adata.layers}
    bdata = AnnData(
        X=all_zeros,
        layers=layers,
        uns=adata.uns.copy(),
        obs=adata.obs,
        var=adata.var,
        varm=adata.varm,
        obsm=adata.obsm,
        obsp=adata.obsp,
    )
    return bdata

class UnpicklerCpu(pickle.Unpickler):
    """Helps to pickle.load a model trained on GPU to CPU.
    
    See also https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
