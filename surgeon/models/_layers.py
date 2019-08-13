from keras import backend as K
from keras.engine import Layer


class SliceLayer(Layer):
    def __init__(self, index=0, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class ColwiseMultLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[0] * K.reshape(x[1], (-1, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


LAYERS = {
    "SliceLayer": SliceLayer,
    "ColWiseMultLayer": ColwiseMultLayer,
}
