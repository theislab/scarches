import pytest
import torch
from scarches.dataset import trVAEDataset
from scanpy.datasets import pbmc3k_processed, pbmc3k

@pytest.mark.parametrize('get_adata', [pbmc3k_processed, pbmc3k])
def test_dataloader(get_adata):
    adata = get_adata()
    adata.obs['study'] = 'single'
    condition_encoder = {'single': 0}

    ds = trVAEDataset(adata, condition_key='study', condition_encoder=condition_encoder)
    batch = ds[10:100]

    assert set(['x', 'labeled', 'sizefactor', 'batch']).issubset(batch.keys())
    assert torch.allclose(batch['x'].sum(1), batch['sizefactor'])
