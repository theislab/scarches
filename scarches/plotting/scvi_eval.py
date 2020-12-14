from scvi.data import get_from_registry

from scarches.metrics.metrics import entropy_batch_mixing, knn_purity, asw, nmi
from scarches.models import SCVI, SCANVI, TOTALVI
from scarches.trainers import scVITrainer, scANVITrainer, totalTrainer

from scipy.sparse import issparse
import numpy as np
import scanpy as sc
import torch
from typing import Union
from sklearn.metrics import f1_score
import anndata
import matplotlib.pyplot as plt

sc.settings.set_figure_params(dpi=200, frameon=False)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


class SCVI_EVAL:
    def __init__(
            self,
            model: Union[SCVI, SCANVI, TOTALVI],
            adata: anndata.AnnData,
            trainer: Union[scVITrainer, scANVITrainer, totalTrainer] = None,
            cell_type_key: str = None,
            batch_key: str = None,
    ):
        self.outer_model = model
        self.model = model.model
        self.model.eval()

        if trainer is None:
            self.trainer = model.trainer
        else:
            self.trainer = trainer

        self.adata = adata
        self.modified = getattr(model.model, 'encode_covariates', True)
        self.annotated = type(model) is SCANVI
        self.predictions = None
        self.certainty = None
        self.prediction_names = None
        self.class_check = None
        self.post_adata_2 = None

        if trainer is not None:
            if self.trainer.use_cuda:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = next(self.model.parameters()).get_device()

        if issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X
        self.x_tensor = torch.tensor(X, device=self.device)

        self.labels = None
        self.label_tensor = None
        if self.annotated:
            self.labels = get_from_registry(self.adata, "labels").astype(np.int8)
            self.label_tensor = torch.tensor(self.labels, device=self.device)
        self.cell_types = self.adata.obs[cell_type_key].tolist()

        self.batch_indices = get_from_registry(self.adata, "batch_indices").astype(np.int8)
        self.batch_tensor = torch.tensor(self.batch_indices, device=self.device)
        self.batch_names = self.adata.obs[batch_key].tolist()
        self.celltype_enc = [0]*len(self.adata.obs[cell_type_key].unique().tolist())
        for i, cell_type in enumerate(self.adata.obs[cell_type_key].unique().tolist()):
            label = self.adata.obs['_scvi_labels'].unique().tolist()[i]
            self.celltype_enc[label] = cell_type
        self.post_adata = self.latent_as_anndata()

    def latent_as_anndata(self):
        if type(self.outer_model) is TOTALVI:
            latent = self.outer_model.get_latent_representation(self.adata)
        else:
            if self.modified:
                latents = self.model.sample_from_posterior_z(
                    self.x_tensor,
                    y=self.label_tensor,
                    batch_index=self.batch_tensor
                )
            else:
                latents = self.model.sample_from_posterior_z(
                    self.x_tensor,
                    y=self.label_tensor,
                )

            if self.annotated:
                latent = latents.cpu().detach().numpy()
                latent2, _, _ = self.model.encoder_z2_z1(latents, self.label_tensor)
                latent2 = latent2.cpu().detach().numpy()
                post_adata_2 = sc.AnnData(latent2)
                post_adata_2.obs['cell_type'] = self.cell_types
                post_adata_2.obs['batch'] = self.batch_names
                self.post_adata_2 = post_adata_2
            else:
                latent = latents.cpu().detach().numpy()

        post_adata = sc.AnnData(latent)
        post_adata.obs['cell_type'] = self.cell_types
        post_adata.obs['batch'] = self.batch_names
        return post_adata

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_latent(self,
                    show=True,
                    save=False,
                    dir_path=None,
                    n_neighbors=8,
                    predictions=False,
                    in_one=False,
                    colors=None):
        """
        if save:
            if dir_path is None:
                name = 'scanvi_latent.png'
            else:
                name = f'{dir_path}.png'
        else:
            name = False
        """
        if self.model is None:
            print("Not possible if no model is provided")
            return
        if save:
            show = False
            if dir_path is None:
                dir_path = 'scanvi_latent'
        sc.pp.neighbors(self.post_adata, n_neighbors=n_neighbors)
        sc.tl.umap(self.post_adata)
        if in_one:
            color = ['cell_type', 'batch']
            if predictions:
                color.append(['certainty', 'predictions', 'type_check'])
            sc.pl.umap(self.post_adata,
                       color=color,
                       ncols=2,
                       frameon=False,
                       wspace=0.6,
                       show=show,
                       save=f'{dir_path}_complete.png' if dir_path else None)
        else:
            sc.pl.umap(self.post_adata,
                       color=['cell_type'],
                       frameon=False,
                       wspace=0.6,
                       show=show,
                       palette=colors,
                       save=f'{dir_path}_celltypes.png' if dir_path else None)
            sc.pl.umap(self.post_adata,
                       color=['batch'],
                       frameon=False,
                       wspace=0.6,
                       show=show,
                       save=f'{dir_path}_batch.png' if dir_path else None)
            if predictions:
                sc.pl.umap(self.post_adata,
                           color=['predictions'],
                           frameon=False,
                           wspace=0.6,
                           show=show,
                           save=f'{dir_path}_predictions.png' if dir_path else None)
                sc.pl.umap(self.post_adata,
                           color=['certainty'],
                           frameon=False,
                           wspace=0.6,
                           show=show,
                           save=f'{dir_path}_certainty.png' if dir_path else None)
                sc.pl.umap(self.post_adata,
                           color=['type_check'],
                           ncols=2,
                           frameon=False,
                           wspace=0.6,
                           show=show,
                           save=f'{dir_path}_type_check.png' if dir_path else None)

    def plot_history(self, show=True, save=False, dir_path=None):
        if self.trainer is None:
            print("Not possible if no trainer is provided")
            return
        if self.annotated:
            fig, axs = plt.subplots(2, 1)
            elbo_full = self.trainer.history["elbo_full_dataset"]
            x_1 = np.linspace(0, len(elbo_full), len(elbo_full))
            axs[0].plot(x_1, elbo_full, label="Full")

            accuracy_labelled_set = self.trainer.history["accuracy_labelled_set"]
            accuracy_unlabelled_set = self.trainer.history["accuracy_unlabelled_set"]
            if len(accuracy_labelled_set) != 0:
                x_2 = np.linspace(0, len(accuracy_labelled_set), (len(accuracy_labelled_set)))
                axs[1].plot(x_2, accuracy_labelled_set, label="accuracy labelled")
            if len(accuracy_unlabelled_set) != 0:
                x_3 = np.linspace(0, len(accuracy_unlabelled_set), (len(accuracy_unlabelled_set)))
                axs[1].plot(x_3, accuracy_unlabelled_set, label="accuracy unlabelled")
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('ELBO')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            if save:
                if dir_path is None:
                    plt.savefig('scanvi_history.png', bbox_inches='tight')
                else:
                    plt.savefig(f'{dir_path}.png', bbox_inches='tight')
            if show:
                plt.show()
        else:
            fig = plt.figure()
            elbo_train = self.trainer.history["elbo_train_set"]
            elbo_test = self.trainer.history["elbo_test_set"]
            x = np.linspace(0, len(elbo_train), len(elbo_train))
            plt.plot(x, elbo_train, label="train")
            plt.plot(x, elbo_test, label="test")
            plt.ylim(min(elbo_train) - 50, min(elbo_train) + 1000)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            if save:
                if dir_path is None:
                    plt.savefig('scvi_history.png', bbox_inches='tight')
                else:
                    plt.savefig(f'{dir_path}.png', bbox_inches='tight')
            if show:
                plt.show()

    def get_ebm(self, n_neighbors=50, n_pools=50, n_samples_per_pool=100, verbose=True):
        ebm_score = entropy_batch_mixing(
            adata=self.post_adata,
            label_key='batch',
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool
        )
        if verbose:
            print("Entropy of Batchmixing-Score:", ebm_score)
        return ebm_score

    def get_knn_purity(self, n_neighbors=50, verbose=True):
        knn_score = knn_purity(
            adata=self.post_adata,
            label_key='cell_type',
            n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:", knn_score)
        return knn_score

    def get_asw(self):
        asw_score_batch, asw_score_cell_types = asw(adata=self.post_adata, label_key='cell_type',batch_key='batch')
        print("ASW on batch:", asw_score_batch)
        print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_nmi(self):
        nmi_score = nmi(adata=self.post_adata, label_key='cell_type')
        print("NMI score:", nmi_score)
        return nmi_score

    def get_latent_score(self):
        ebm = self.get_ebm(verbose=False)
        knn = self.get_knn_purity(verbose=False)
        score = ebm + knn
        print("Latent-Space Score (KNN + EBM):", score)
        return score

    def get_classification_accuracy(self):
        if self.annotated:
            if self.modified:
                softmax = self.model.classify(self.x_tensor, batch_index=self.batch_tensor)
            else:
                softmax = self.model.classify(self.x_tensor)
            softmax = softmax.cpu().detach().numpy()
            self.predictions = np.argmax(softmax, axis=1)
            self.certainty = np.max(softmax, axis=1)
            self.prediction_names = [0]*self.predictions.shape[0]
            for index, label in np.ndenumerate(self.predictions):
                self.prediction_names[index[0]] = self.celltype_enc[label]
            self.class_check = np.array(np.expand_dims(self.predictions, axis=1) == self.labels)
            class_check_labels = [0] * self.class_check.shape[0]
            for index, check in np.ndenumerate(self.class_check):
                class_check_labels[index[0]] = 'Correct' if check else 'Incorrect'
            accuracy = np.sum(self.class_check) / self.class_check.shape[0]
            self.post_adata.obs['certainty'] = self.certainty
            self.post_adata.obs['type_check'] = class_check_labels
            self.post_adata.obs['predictions'] = self.prediction_names
            print("Classification Accuracy: %0.2f" % accuracy)
            return accuracy
        else:
            print("Classification ratio not available for scVI models")

    def get_f1_score(self):
        if self.annotated:
            if self.modified:
                predictions = self.model.classify(self.x_tensor, batch_index=self.batch_tensor)
            else:
                predictions = self.model.classify(self.x_tensor)
            self.predictions = predictions.cpu().detach().numpy()
            self.predictions = np.expand_dims(np.argmax(self.predictions, axis=1), axis=1)
            score = f1_score(self.labels, self.predictions, average='macro')
            print("F1 Score: %0.2f" % score)
            return score
        else:
            print("F1 Score not available for scVI models")
