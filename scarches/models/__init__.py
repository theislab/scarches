from .trvae.trvae import trVAE
from .trvae.trvae_model import TRVAE
from .trvae.adaptors import Adaptor, attach_adaptors
from .scgen.vaearith import vaeArith
from .scgen.vaearith_model import scgen

import logging
try:
    from .sagenet.sagenet import sagenet
    from .sagenet.utils import glasso
except:
    logging.warning('In order to use sagenet models, please install pytorch geometric (see https://pytorch-geometric.readthedocs.io) and \n captum (see https://github.com/pytorch/captum).')
from .expimap.expimap import expiMap
from .expimap.expimap_model import EXPIMAP
from scvi.model import SCVI, SCANVI, TOTALVI
try:
    import tcr_embedding as mvTCR
except:
    logging.warning('mvTCR is not installed. To use mvTCR models, please install it first using "pip install mvtcr"')
try:
    from multigrate.data import organize_multiome_anndatas
    from multigrate.model import MultiVAE
except:
    logging.warning('multigrate is not installed. To use multigrate models, please install it first using "pip install multigrate".')
