import numpy as np


class EarlyStopping(object):
    """Class for EarlyStopping. This class contains the implementation of early stopping for TRVAE/CVAE training.

           Parameters
           ----------
           early_stopping_metric: : String
                The metric/loss which the early stopping criterion gets calculated on.
           threshold: Float
                The minimum value which counts as improvement.
           patience: Integer
                Number of epochs which are allowed to have no improvement until the training is stopped.
           reduce_lr: Boolean
                If 'True', the learning rate gets adjusted by 'lr_factor' after a given number of epochs with no
                improvement.
           lr_patience: Integer
                Number of epochs which are allowed to have no improvement until the learning rate is adjusted.
           lr_factor: Float
                Scaling factor for adjusting the learning rate.
        """
    def __init__(self,
                 early_stopping_metric: str = 'val_unweighted_loss',
                 threshold: float = 0,
                 patience: int = 20,
                 reduce_lr: bool = True,
                 lr_patience: int = 7,
                 lr_factor: float = 0.1):

        self.early_stopping_metric = early_stopping_metric
        self.threshold = threshold
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.epoch = 0
        self.wait = 0
        self.wait_lr = 0
        self.current_performance = np.inf
        self.best_performance = np.inf
        self.best_performance_state = np.inf

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, scalar):
        self.epoch += 1
        if self.epoch < self.patience:
            continue_training = True
            lr_update = False
        elif self.wait >= self.patience:
            continue_training = False
            lr_update = False
        else:
            if not self.reduce_lr:
                lr_update = False
            elif self.wait_lr >= self.lr_patience:
                lr_update = True
                self.wait_lr = 0
            else:
                lr_update = False
            # Shift
            self.current_performance = scalar

            improvement = self.best_performance - self.current_performance

            # updating best performance
            if improvement > 0:
                self.best_performance = self.current_performance

            if improvement < self.threshold:
                self.wait += 1
                self.wait_lr += 1
            else:
                self.wait = 0
                self.wait_lr = 0

            continue_training = True

        if not continue_training:
            print("\nStopping early: no improvement of more than " + str(self.threshold) +
                  " nats in " + str(self.patience) + " epochs")
            print("If the early stopping criterion is too strong, "
                  "please instantiate it with different parameters in the train method.")
        return continue_training, lr_update

    def update_state(self, scalar):
        improved = (self.best_performance_state - scalar) > 0
        if improved:
            self.best_performance_state = scalar
        return improved
