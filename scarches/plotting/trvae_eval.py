import numpy as np
import scanpy as sc
import torch
import anndata
import matplotlib.pyplot as plt
from typing import Union

from scarches.dataset.trvae._utils import label_encoder
from scarches.metrics.metrics import entropy_batch_mixing, knn_purity, asw, nmi
from scarches.models import trVAE, TRVAE
from scarches.trainers import trVAETrainer

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


class TRVAE_EVAL:
    def __init__(
            self,
            model: Union[trVAE, TRVAE],
            adata: anndata.AnnData,
            trainer: trVAETrainer = None,
            condition_key: str = None,
            cell_type_key: str = None
    ):
        if type(model) is TRVAE:
            trainer = model.trainer
            model = model.model

        self.model = model
        self.trainer = trainer
        self.adata = adata
        self.device = model.device
        self.conditions, _ = label_encoder(
            self.adata,
            encoder=model.condition_encoder,
            condition_key=condition_key,
        )
        self.cell_type_names = None
        self.batch_names = None
        if cell_type_key is not None:
            self.cell_type_names = adata.obs[cell_type_key].tolist()
        if condition_key is not None:
            self.batch_names = adata.obs[condition_key].tolist()

        self.adata_latent = self.latent_as_anndata()

    def latent_as_anndata(self):
        if self.model.calculate_mmd == 'z' or self.model.use_mmd == False:
            latent = self.model.get_latent(
                self.adata.X,
                c=self.conditions,
            )
        else:
            latent = self.model.get_y(
                self.adata.X,
                c=self.conditions
            )
        adata_latent = sc.AnnData(latent)
        if self.cell_type_names is not None:
            adata_latent.obs['cell_type'] = self.cell_type_names
        if self.batch_names is not None:
            adata_latent.obs['batch'] = self.batch_names
        return adata_latent

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_latent(self,
                    show=True,
                    save=False,
                    dir_path=None,
                    n_neighbors=8,
                    ):
        if save:
            show=False
            if dir_path is None:
                save = False

        sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
        sc.tl.umap(self.adata_latent)
        color = [
            'cell_type' if self.cell_type_names is not None else None,
            'batch' if self.batch_names is not None else None,
        ]
        sc.pl.umap(self.adata_latent,
                   color=color,
                   frameon=False,
                   wspace=0.6,
                   show=show)
        if save:
            plt.savefig(f'{dir_path}_batch.png', bbox_inches='tight')

    def plot_history(self, show=True, save=False, dir_path=None):
        if save:
            show = False
            if dir_path is None:
                save = False

        if self.trainer is None:
            print("Not possible if no trainer is provided")
            return
        fig = plt.figure()
        elbo_train = self.trainer.logs["epoch_loss"]
        elbo_test = self.trainer.logs["val_loss"]
        x = np.linspace(0, len(elbo_train), num=len(elbo_train))
        plt.plot(x, elbo_train, label="Train")
        plt.plot(x, elbo_test, label="Validate")
        plt.ylim(min(elbo_test) - 50, max(elbo_test) + 50)
        plt.legend()
        if save:
            plt.savefig(f'{dir_path}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()

    def get_ebm(self, n_neighbors=50, n_pools=50, n_samples_per_pool=100, verbose=True):
        ebm_score = entropy_batch_mixing(
            adata=self.adata_latent,
            label_key='batch',
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool
        )
        if verbose:
            print("Entropy of Batchmixing-Score: %0.2f" % ebm_score)
        return ebm_score

    def get_knn_purity(self, n_neighbors=50, verbose=True):
        knn_score = knn_purity(
            adata=self.adata_latent,
            label_key='cell_type',
            n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:  %0.2f" % knn_score)
        return knn_score

    def get_asw(self):
        asw_score_batch, asw_score_cell_types = asw(adata=self.adata_latent, label_key='cell_type', batch_key='batch')
        print("ASW on batch:", asw_score_batch)
        print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_nmi(self):
        nmi_score = nmi(adata=self.adata_latent, label_key='cell_type')
        print("NMI score:", nmi_score)
        return nmi_score

    def get_latent_score(self):
        ebm = self.get_ebm(verbose=False)
        knn = self.get_knn_purity(verbose=False)
        score = ebm + knn
        print("Latent-Space Score EBM+KNN, EBM, KNN: %0.2f, %0.2f, %0.2f" % (score, ebm, knn))
        return score
