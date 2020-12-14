import scanpy as sc
from scipy import sparse


def remove_sparsity(adata):
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        return new_adata

    return adata
