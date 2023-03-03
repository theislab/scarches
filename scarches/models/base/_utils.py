import numpy as np
import logging
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
        if isinstance(adata, csr_matrix): # support sparse matrix
            filling_X = csr_matrix(filling_X)
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
