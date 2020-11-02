import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import sys


def compute_kernel(x, y, kernel='rbf', **kwargs):
    """
        Computes RBF kernel between x and y.
        # Parameters
            x: Tensor
                Tensor with shape [batch_size, z_dim]
            y: Tensor
                Tensor with shape [batch_size, z_dim]
        # Returns
            returns the computed RBF kernel between x and y
    """
    scales = kwargs.get("scales", [])
    if kernel == "rbf":
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
    elif kernel == 'raphy':
        scales = K.variable(value=np.asarray(scales))
        squared_dist = K.expand_dims(squared_distance(x, y), 0)
        scales = K.expand_dims(K.expand_dims(scales, -1), -1)
        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        weights = K.expand_dims(K.expand_dims(weights, -1), -1)
        return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
    elif kernel == "multi-scale-rbf":
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
        distances = squared_distance(x, y)
        s = K.dot(beta, K.reshape(distances, (1, -1)))

        return K.reshape(tf.reduce_sum(tf.exp(-s), 0), K.shape(distances)) / len(sigmas)


def squared_distance(x, y):  # returns the pairwise euclidean distance
    r = K.expand_dims(x, axis=1)
    return K.sum(K.square(r - y), axis=-1)


def compute_mmd(x, y, kernel, **kwargs):  # [batch_size, z_dim] [batch_size, z_dim]
    """
        Computes Maximum Mean Discrepancy(MMD) between x and y.
        # Parameters
            x: Tensor
                Tensor with shape [batch_size, z_dim]
            y: Tensor
                Tensor with shape [batch_size, z_dim]
        # Returns
            returns the computed MMD between x and y
    """
    x_kernel = compute_kernel(x, x, kernel=kernel, **kwargs)
    y_kernel = compute_kernel(y, y, kernel=kernel, **kwargs)
    xy_kernel = compute_kernel(x, y, kernel=kernel, **kwargs)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


def sample_z(args):
    """
        Samples from standard Normal distribution with shape [size, z_dim] and
        applies re-parametrization trick. It is actually sampling from latent
        space distributions with N(mu, var) computed in `_encoder` function.
        # Parameters
            No parameters are needed.
        # Returns
            The computed Tensor of samples with shape [size, z_dim].
    """
    mu, log_var = args
    batch_size = K.shape(mu)[0]
    z_dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=[batch_size, z_dim])
    return mu + K.exp(log_var / 2) * eps


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def print_message(epoch, logs, n_epochs=10000, duration=50):
    if epoch % duration == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}:")
        print(
            f" - loss: {logs['loss']:.4f} - reconstruction_loss: {logs['reconstruction_loss']:.4f} - mmd_loss: {logs['mmd_loss']:.4f}"
            f" - val_loss: {logs['val_loss']:.4f}"
            f" - val_reconstruction_loss: {logs['val_reconstruction_loss']:.4f} - val_mmd_loss: {logs['val_mmd_loss']:.4f}")


def print_progress(epoch, logs, n_epochs=10000):
    message = f' - loss: {logs["loss"]:.4f}'
    train_keys = [key for key in sorted(list(logs.keys())) if (not key.startswith('val_') and key != 'loss')]

    for key in train_keys:
        message += f' - {key}: {logs[key]:.4f}'

    message += f' - val_loss: {logs["val_loss"]:.4f}'
    valid_keys = [key for key in sorted(list(logs.keys())) if (key.startswith('val_') and key != 'val_loss')]

    for key in valid_keys:
        message += f' - {key}: {logs[key]:.4f}'

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)


def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + '-' * (length - filled_len)
    sys.stdout.write(str(f'\r{prefix} |{bar}| {percent}% {suffix}'.encode("utf-8"), "utf-8")),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
