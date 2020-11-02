import tensorflow as tf
import numpy as np
from scipy import sparse

from scarches.utils import label_encoder


def preprocess_cvae_input(n_conditions):
    def preprocess(x, y):
        x['encoder_label'] = tf.reshape(tf.one_hot(tf.cast(x['encoder_label'], tf.uint8), n_conditions), (n_conditions,))
        x['decoder_label'] = tf.reshape(tf.one_hot(tf.cast(x['decoder_label'], tf.uint8), n_conditions), (n_conditions,))

        return x, y

    return preprocess


def make_dataset(adata, condition_key, le, batch_size, n_epochs, is_training, loss_fn, n_conditions,
                 size_factor_key=None, use_mmd=False):
    if sparse.issparse(adata.X):
        expressions = adata.X.A
    else:
        expressions = adata.X

    encoded_conditions, le = label_encoder(adata, le, condition_key)
    if loss_fn == 'nb':
        if sparse.issparse(adata.raw.X):
            raw_expressions = adata.raw.X.A
        else:
            raw_expressions = adata.raw.X
        dataset = tf.data.Dataset.from_tensor_slices(({"expression": expressions,
                                                       "encoder_label": encoded_conditions,
                                                       "decoder_label": encoded_conditions,
                                                       "size_factor": adata.obs[size_factor_key].values},
                                                      {"reconstruction": raw_expressions}
                                                      ))
    elif loss_fn == 'zinb':
        if sparse.issparse(adata.raw.X):
            raw_expressions = adata.raw.X.A
        else:
            raw_expressions = adata.raw.X
        dataset = tf.data.Dataset.from_tensor_slices(({"expression": expressions,
                                                       "encoder_label": encoded_conditions,
                                                       "decoder_label": encoded_conditions,
                                                       'size_factor': adata.obs[size_factor_key].values},
                                                      {"reconstruction": raw_expressions}
                                                      ))
    else:
        if use_mmd:
            dataset = tf.data.Dataset.from_tensor_slices(({"expression": expressions,
                                                           "encoder_label": encoded_conditions,
                                                           "decoder_label": encoded_conditions},
                                                          {"reconstruction": expressions,
                                                           "mmd": encoded_conditions}
                                                          ))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({"expression": expressions,
                                                           "encoder_label": encoded_conditions,
                                                           "decoder_label": encoded_conditions},
                                                          {"reconstruction": expressions}
                                                          ))
    if is_training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(preprocess_cvae_input(n_conditions),
                          num_parallel_calls=4,
                          deterministic=None)
    if is_training:
        dataset = dataset.repeat(n_epochs)
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
    dataset = dataset.prefetch(buffer_size=5)

    return dataset, le
