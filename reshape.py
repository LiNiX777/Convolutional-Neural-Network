import numpy as np
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
        output = np.zeros((self.channels, self.output_height, self.output_width))
        for i in range(self.channels):
            for a in range(self.output_height):
                for b in range(self.output_width):
                    h, w = a * self.stride, b * self.stride
                    output[i,a,b] = np.max(input[i, h:h+self.pool_height, w:w+self.pool_width])
        return output
        

    def backpropagation(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)

        for i in range(self.channels):
            for a in range(self.output_height):
                for b in range(self.output_width):
                    h, w = a * self.stride, b * self.stride
                    window = self.input[i, h:h+self.pool_height, w:w+self.pool_width]
                    mask = (window == np.max(window))

                    input_gradient[i, h:h+self.pool_height, w:w+self.pool_width] += mask * output_gradient[i, a, b]
        return input_gradient


class Flatten_layer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_pass(self, input):
        return np.reshape(input, self.output_shape)

    def backpropagation(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

