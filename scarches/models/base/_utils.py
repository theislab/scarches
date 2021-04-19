import numpy as np
import logging

logger = logging.getLogger(__name__)


def _validate_var_names(adata, source_var_names):
    user_var_names = adata.var_names.astype(str)
    if not np.array_equal(source_var_names, user_var_names):
        logger.warning(
            "var_names for adata passed in does not match var_names of "
            "adata used to train the model. For valid results, the vars "
            "need to be the same and in the same order as the adata used to train the model."
        )
