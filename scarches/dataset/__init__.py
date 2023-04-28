from .trvae._utils import label_encoder
from .trvae.data_handling import remove_sparsity
from .trvae.anndata import AnnotatedDataset as trVAEDataset
from .scpoli.anndata import MultiConditionAnnotatedDataset

import logging
try:
    from .MGA_data import seqFISH1_1, seqFISH2_1, seqFISH3_1, seqFISH1_2, seqFISH2_2, seqFISH3_2, scRNAseq
except:
    logging.warning('In order to use the mouse gastrulation seqFISH datsets, please install squidpy (see https://github.com/scverse/squidpy).')

__all__ = ('label_encoder', 'remove_sparsity', 'trVAEDataset')
