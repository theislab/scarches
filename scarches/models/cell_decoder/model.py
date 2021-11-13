from scvi.model.base import BaseModelClass, VAEMixin,UnsupervisedTrainingMixin
from scvi.data import register_tensor_from_anndata
from scvi.data._anndata import _setup_anndata
from typing import Optional
from anndata import AnnData
import numpy as np

from .module import CellDecoderVAE

class CellDecoder(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(self, adata: AnnData, **model_kwargs):
        adata.obs["_indices"] = np.arange(adata.n_obs)
        register_tensor_from_anndata(adata, "ind_x", "obs", "_indices")

        super(CellEncoder, self).__init__(adata)

        n_batch = self.summary_stats["n_batch"]

        self.module = CellDecoderVAE(
            n_obs=self.summary_stats["n_cells"],
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            **model_kwargs
        )

        self.init_params_ = self._get_init_params(locals())

    @staticmethod
    def setup_anndata(
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:
        return _setup_anndata(
            adata,
            batch_key=batch_key,
            layer=layer,
            copy=copy
        )
