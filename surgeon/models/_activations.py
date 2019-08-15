import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, Lambda
from keras.layers.advanced_activations import LeakyReLU


def mean_activation(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def disp_activation(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


ACTIVATIONS = {
    "relu": Activation("relu", name='reconstruction_output'),
    'leaky_relu': LeakyReLU(name="reconstruction_output"),
    'linear': Activation("linear", name='reconstruction_output'),
    'mean_activation': Lambda(lambda x: mean_activation(x), name="decoder_mean"),
    'disp_activation': Lambda(lambda x: disp_activation(x), name="decoder_disp"),
    'sigmoid': Activation('sigmoid', name='decoder_pi'),
}
