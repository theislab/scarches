import scanpy as sc


def pancreas():
    url = f"https://zenodo.org/record/3930949/files/pancreas_normalized.h5ad?download=1"
    adata = sc.read("./data/", backup_url=url, sparse=True, cache=True)
    adata.var_names_make_unique()
    return adata
