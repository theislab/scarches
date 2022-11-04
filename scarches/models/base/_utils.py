import numpy as np
import logging
from anndata import AnnData

logger = logging.getLogger(__name__)


def _validate_var_names(adata, source_var_names):
    user_var_names = adata.var_names.astype(str)
    new_adata = adata.copy()

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
        new_target_X = np.concatenate((adata.X, filling_X), axis=1)
        new_target_vars = adata.var_names.tolist() + ref_genes_not_in_query
        new_adata = AnnData(new_target_X)
        new_adata.var_names = new_target_vars
        new_adata.obs = adata.obs.copy()

    if len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)) > 0:
        print(
            "Query data contains expression data of ",
            len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)),
            " genes that were not contained in the reference dataset. This information "
            "will be removed from the query data object for further processing.")

    #Warning for gene percentage   
    try:
        query_genes = adata.var_names
        ref_genes = source_var_names
        percentage = round((query_genes.intersection(ref_genes).size/query_genes.size)*100,4)
        if percentage != 100:
            logging.warning(f"WARNING: Query shares {percentage}% of its genes with the reference. This may lead to inaccuracy in the results.")
    except Exception:
            logging.warning("WARNING: Something is wrong with the reference genes.")

    # remove unseen gene information and order anndata
    new_adata = new_adata[:, source_var_names].copy()

    print(new_adata)

    return new_adata
