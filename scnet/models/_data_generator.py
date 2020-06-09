import keras
import numpy as np
from scipy.sparse import issparse


def desparse(data):
    if issparse(data):
        data = data.A
    return data


class UnsupervisedDataGenerator(keras.utils.Sequence):
    def __init__(self, adata, encoded_conditions, use_raw=False, n_conditions=1,
                 batch_size=32,
                 shuffle=True):
        self.adata = adata
        self.encoded_conditions = encoded_conditions
        self.batch_size = batch_size
        self.n_conditions = n_conditions
        self.use_raw = use_raw
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, index):
        start = index
        end = index + self.batch_size
        indexes = self.indexes[start:end]

        expression = desparse(self.adata.X[indexes])
        encoded_condition = self.encoded_conditions[indexes]
        one_hot_condition = keras.utils.to_categorical(encoded_condition, num_classes=self.n_conditions)

        if self.use_raw:
            target_expression = desparse(self.adata.raw.X[indexes])
        else:
            target_expression = expression

        X = [expression, one_hot_condition, one_hot_condition]
        y = [target_expression, encoded_condition]

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.adata))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class SupervisedDataGenerator(keras.utils.Sequence):
    def __init__(self, adata, encoded_conditions, encoded_labels,
                 use_raw=False, n_conditions=1, n_cell_types=1,
                 batch_size=32,
                 shuffle=True):
        self.adata = adata
        self.encoded_conditions = encoded_conditions
        self.encoded_cell_types = encoded_labels
        self.batch_size = batch_size
        self.n_conditions = n_conditions
        self.n_cell_types = n_cell_types
        self.use_raw = use_raw
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, index):
        start = index
        end = index + self.batch_size
        indexes = self.indexes[start:end]

        expression = desparse(self.adata.X[indexes])
        encoded_condition = self.encoded_conditions[indexes]
        encoded_cell_type = self.encoded_cell_types[indexes]
        one_hot_condition = keras.utils.to_categorical(encoded_condition, num_classes=self.n_conditions)
        one_hot_cell_type = keras.utils.to_categorical(encoded_cell_type, num_classes=self.n_cell_types)

        if self.use_raw:
            target_expression = desparse(self.adata.raw.X[indexes])
        else:
            target_expression = expression

        X = [expression, one_hot_condition, one_hot_condition]
        y = [target_expression, encoded_condition, one_hot_cell_type]

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.adata))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
