import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from layer import Layer

class Maxpool_layer(Layer):
    def __init__(self, input_shape, pool_shape, stride):
        self.channels, self.height, self.width = input_shape
        self.pool_height, self.pool_width = pool_shape
        self.stride = stride
        self.output_height = ((self.height - self.pool_height) // self.stride + 1)
        self.output_width = ((self.width - self.pool_width) // self.stride + 1)
        
    def forward_pass(self, input):
        self.input = input
        self.windows = sliding_window_view(input, (self.pool_height, self.pool_width), axis=(1, 2))[:, ::self.stride, ::self.stride]
        self.output = np.max(self.windows, axis=(3, 4))
        return self.output
        
    def backpropagation(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)
        mask = (self.windows == self.output[:, :, :, None, None])
        gradient = mask * output_gradient[:, :, :, None, None]

        for a in range(self.pool_height):
            for b in range(self.pool_width):
                h_end = a + self.stride * self.output_height
                w_end = b + self.stride * self.output_width
                input_gradient[:, a:h_end:self.stride, b:w_end:self.stride] += gradient[:, :, :, a, b]
        return input_gradient   

class Flatten_layer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_pass(self, input):
        return np.reshape(input, self.output_shape)

    def backpropagation(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)