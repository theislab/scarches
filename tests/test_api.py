import scarches as sca
import numpy as np
import scanpy as sc


def test_api():
    # reference adata
    X = np.random.normal(size=(1000, 10))
    adata = sc.AnnData(X=X)
    adata.obs['cell_type'] = list("ABCDE") * 200
    adata.obs['condition'] = list("MNOP") * 250
    adata.raw = adata
    adata.obs['size_factors'] = 1.0

    # print(len(adata.obs['condition'].unique().tolist()))
    # print(len(adata.obs['cell_type'].unique().tolist()))

    model = sca.models.scArches(10, list('MNOP'))
    model.train(adata, "condition", 0.8)

    print(model.network_kwargs)
    # query adata
    X = np.random.normal(size=(1000, 10))
    adata = sc.AnnData(X=X)
    adata.obs['cell_type'] = list("ABCL") * 250
    adata.obs['condition'] = list("QRST") * 250
    adata.raw = adata
    adata.obs['size_factors'] = 1.0

    new_model = sca.operate(model, "new_task", adata.obs['condition'].unique().tolist())
    print(new_model.network_kwargs)
    # print(new_model.n_conditions, new_model.n_mmd_conditions, new_model.condition_encoder)
    new_model.train(adata, "condition", 0.8, n_epochs=1)
