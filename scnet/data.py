import scanpy as sc
from scipy import sparse
import numpy as np


def read(filename, **kwargs):
    """Reads the dataset. For more information about this function, read `here <scanpy.readthedocs.io>`_.

        Parameters
        ----------
        filename: str
            path to the dataset.
        kwargs:
            Other ``scanpy.read`` function arguments.

        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.

    """
    return sc.read(filename, **kwargs)


def normalize_hvg(adata, batch_key=None, size_factors=True, logtrans_input=True,
                  target_sum=None, n_top_genes=2000):
    """Normalizes, and select highly variable genes of ``adata``.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        batch_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame. This is
            used for selecting highly variable genes.
        size_factors: bool
            whether to add size factors (or n_counts) as a column after normalization in ``adata.obs`` matrix or not.
            if `True`, the added column name is ``size_factors``.
        logtrans_input: bool
            If ``True``, will apply ``scanpy.pp.log1p`` function on ``adata.X`` after normalizing per cell.
        target_sum: float
            If ``None``, after normalization, each observation (cell) has a total count
            equal to the median of total counts for observations (cells)
            before normalization.
        n_top_genes: int
            Number of highly variable genes to be selected after normalization.

        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Normalized and hvg selected annotated dataset.

    """

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=True, key_added='size_factors')
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        if batch_key:
            genes = _hvg_batch(adata.copy(), batch_key=batch_key, adataOut=False, target_genes=n_top_genes)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


def _hvg_batch(adata, batch_key=None, target_genes=2000, flavor='cell_ranger', n_bins=20, adataOut=False):
    """
    Method to select HVGs based on mean dispersions of genes that are highly
    variable genes in all batches. Using a the top target_genes per batch by
    average normalize dispersion. If target genes still hasn't been reached,
    then HVGs in all but one batches are used to fill up. This is continued
    until HVGs in a single batch are considered.
    """

    adata_hvg = adata if adataOut else adata.copy()

    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    # Calculate double target genes per dataset
    sc.pp.highly_variable_genes(adata_hvg,
                                flavor=flavor,
                                n_top_genes=target_genes,
                                n_bins=n_bins,
                                batch_key=batch_key)

    nbatch1_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches >
                                                            len(adata_hvg.obs[batch_key].cat.categories) - 1]

    nbatch1_dispersions.sort_values(ascending=False, inplace=True)

    if len(nbatch1_dispersions) > target_genes:
        hvg = nbatch1_dispersions.index[:target_genes]

    else:
        enough = False
        print(f'Using {len(nbatch1_dispersions)} HVGs from full intersect set')
        hvg = nbatch1_dispersions.index[:]
        not_n_batches = 1

        while not enough:
            target_genes_diff = target_genes - len(hvg)

            tmp_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches ==
                                                                (n_batches - not_n_batches)]

            if len(tmp_dispersions) < target_genes_diff:
                print(f'Using {len(tmp_dispersions)} HVGs from n_batch-{not_n_batches} set')
                hvg = hvg.append(tmp_dispersions.index)
                not_n_batches += 1

            else:
                print(f'Using {target_genes_diff} HVGs from n_batch-{not_n_batches} set')
                tmp_dispersions.sort_values(ascending=False, inplace=True)
                hvg = hvg.append(tmp_dispersions.index[:target_genes_diff])
                enough = True

    print(f'Using {len(hvg)} HVGs')

    if not adataOut:
        del adata_hvg
        return hvg.tolist()
    else:
        return adata_hvg[:, hvg].copy()


def subsample(adata, batch_key, fraction=0.1, specific_cell_types=None, cell_type_key=None):
    """
        Performs Stratified subsampling on ``adata`` while keeping all samples for cell types in ``specific_cell_types``
        if passed.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        batch_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        fraction: float
            Fraction of cells out of all cells in each study to be subsampled.
        specific_cell_types: list
            if `None` will just subsample based on ``batch_key`` in a stratified way. Otherwise, will keep all samples
            with specific cell types in the list and do not subsample them.
        cell_type_key: str
            Name of the column which contains information about different cell types in ``adata.obs`` data frame.

        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Subsampled annotated dataset.

    """
    studies = adata.obs[batch_key].unique().tolist()
    if specific_cell_types and cell_type_key:
        subsampled_adata = adata[adata.obs[cell_type_key].isin(specific_cell_types)]
        other_adata = adata[~adata.obs[cell_type_key].isin(specific_cell_types)]
    else:
        subsampled_adata = None
        other_adata = adata
    for study in studies:
        study_adata = other_adata[other_adata.obs[batch_key] == study]
        n_samples = study_adata.shape[0]
        subsample_idx = np.random.choice(n_samples, int(fraction * n_samples), replace=False)
        study_adata_subsampled = study_adata[subsample_idx, :]
        subsampled_adata = study_adata_subsampled if subsampled_adata is None else subsampled_adata.concatenate(
            study_adata_subsampled)
    return subsampled_adata
