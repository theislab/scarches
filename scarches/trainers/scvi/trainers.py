from typing import Union
from scvi.core.trainers import UnsupervisedTrainer, SemiSupervisedTrainer, TotalTrainer


class scVITrainer(UnsupervisedTrainer):
    def __init__(
        self,
        scvi_model,
        train_size: Union[int, float] = 0.9,
        test_size: Union[int, float] = None,
        *args,
        **kwargs
    ):
        model = scvi_model.model
        adata = scvi_model.adata

        train_size = float(train_size)
        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError(
                "train_size needs to be greater than 0 and less than or equal to 1"
            )

        super().__init__(model, adata, train_size, test_size, *args, **kwargs)

        self.train_set, self.test_set, self.validation_set = self.train_test_validation(
            model, adata, train_size, test_size
        )
        self.train_set.to_monitor = ["elbo"]
        self.test_set.to_monitor = ["elbo"]
        self.validation_set.to_monitor = ["elbo"]
        self.n_samples = len(self.train_set.indices)

        self._scvi_model = scvi_model

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self._scvi_model.trainer = self
        self._scvi_model.is_trained_ = True


class scANVITrainer(SemiSupervisedTrainer):
    def __init__(self, scanvi_model, n_labelled_samples_per_class=50, *args, **kwargs):
        model = scanvi_model.model
        adata = scanvi_model.adata

        super().__init__(model, adata, n_labelled_samples_per_class, *args, **kwargs)

        self._scanvi_model = scanvi_model
        self._labelled = n_labelled_samples_per_class > 0

    def loss(self, tensors_all, tensors_labelled=None):
        if self._labelled:
            return SemiSupervisedTrainer.loss(self, tensors_all, tensors_labelled)
        else:
            return UnsupervisedTrainer.loss(self, tensors_all)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        self._scanvi_model.trainer = self
        self._scanvi_model.is_trained_ = True

    @property
    def scvi_data_loaders_loop(self):
        if self._labelled:
            return ["full_dataset", "labelled_set"]
        else:
            return ["full_dataset"]


class totalTrainer(TotalTrainer):
    def __init__(self, totalvi_model, *args, **kwargs):
        model = totalvi_model.model
        adata = totalvi_model.adata

        # adapive use_adversarial_loss
        if len(args) < 7:
            if 'use_adversarial_loss' not in kwargs:
                if 'totalvi_batch_mask' in totalvi_model.scvi_setup_dict_:
                    imputation = True
                else:
                    imputation = False
                kwargs['use_adversarial_loss'] = imputation

        super().__init__(model, adata, *args, **kwargs)

        self._totalvi_model = totalvi_model

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        self._totalvi_model.trainer = self
        self._totalvi_model.is_trained_ = True
        self._totalvi_model.train_indices_ = self.train_set.indices
        self._totalvi_model.test_indices_ = self.test_set.indices
        self._totalvi_model.validation_indices_ = self.validation_set.indices
