import anndata as ad
from typing import Optional
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res

def reads_to_fragments(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
):
    """
    Function to convert read counts to appoximate fragment counts
    Parameters
    ----------
    adata
        AnnData object that contains read counts.
    layer
        Layer that the read counts are stored in.
    key_added
        Name of layer where the fragment counts will be stored. 
    copy
        Whether to modify copied input object. 

    This function was taken from  Martens et al. 2022,  
    https://github.com/theislab/scatac_poisson_reproducibility
    """ 
    if copy:
        adata = adata.copy()
        
    if layer:
        data = np.ceil(adata.layers[layer].data/2)
    else:
        data = np.ceil(adata.X.data/2)
    
    if key_added:
        adata.layers[key_added] = adata.X.copy()
        adata.layers[key_added].data = data
    elif layer and key_added is None:
        adata.layers[layer].data = data
    elif layer is None and key_added is None:
        adata.X.data = data
    if copy:
        return adata  