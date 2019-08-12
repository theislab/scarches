import tensorflow as tf
from keras import backend as K

from surgeon.models._utils import compute_mmd


def kl_recon(mu, log_var, alpha=0.1, eta=1.0):
    def kl_recon_loss(y_true, y_pred):
        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
        return eta * recon_loss + alpha * kl_loss

    return kl_recon_loss


def mmd(n_conditions, beta, kernel_method='multi-scale-rbf'):
    def mmd_loss(real_labels, y_pred):
        with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
            real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
            conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=n_conditions)
            loss = 0.0
            for i in range(len(conditions_mmd)):
                for j in range(i):
                    loss += compute_mmd(conditions_mmd[j], conditions_mmd[j + 1], kernel_method)
            return beta * loss

    return mmd_loss
