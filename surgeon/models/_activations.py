import tensorflow as tf
from keras import backend as K
from keras.layers import ReLU, Activation
from keras.layers.advanced_activations import LeakyReLU


def mean_activation(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def disp_activation(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


ACTIVATIONS = {
    "relu": ReLU,
    'leaky_relu': LeakyReLU,
    'linear': Activation("linear"),
    'mean_activation': mean_activation,
    'disp_activation': disp_activation,
    'sigmoid': Activation('sigmoid'),
}
