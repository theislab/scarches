import torch.nn as nn

ACTIVATIONS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'linear': nn.Identity(),
    'sigmoid': nn.Sigmoid(),
    'softmax': nn.Softmax(),
    'log_softmax': nn.LogSoftmax(),
    'elu': nn.ELU(),
}