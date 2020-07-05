import scanpy as sc


def pancreas():
    """
    Automatically downloads 5 pancreatic normalized datasets with 1000 selected highly variable genes.
    """
    url = f"https://zenodo.org/record/3930949/files/pancreas_normalized.h5ad?download=1"
    adata = sc.read("./data/pancreas_normalized_hvg.h5ad", backup_url=url, sparse=True, cache=True)
    adata.var_names_make_unique()
    return adata
