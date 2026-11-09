import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from layer import Layer

class Convolutional_layer(Layer):
    def __init__(self, input_shape, kernel_shape, kernel_num):
        self.channels, self.height, self.width = input_shape
        self.kernel_height, self.kernel_width = kernel_shape
        self.kernel_num = kernel_num
        self.output_height = self.height - self.kernel_height + 1
        self.output_width = self.width - self.kernel_width + 1
        self.window_size = self.channels * self.kernel_height * self.kernel_width
        self.kernel = np.random.randn(self.kernel_num, self.channels, self.kernel_height, self.kernel_width) * 0.1
        self.bias = np.random.randn(self.kernel_num) * 0.1

    def forward_pass(self, input):
        self.input = input
        windows = sliding_window_view(input, (self.kernel_height, self.kernel_width), axis=(1, 2))
        self.rows = windows.transpose(1, 2, 0, 3, 4).reshape(-1, self.window_size)
        kernels = self.kernel.reshape(self.kernel_num, self.window_size)

        output = np.dot(kernels, self.rows.T).reshape(self.kernel_num, self.output_height, self.output_width)
        return output + self.bias[:, None, None]

    def backpropagation(self, output_gradient, learning_rate):
        gradient = output_gradient.reshape(self.kernel_num, -1)
        kernels = self.kernel.reshape(self.kernel_num, self.window_size)

        kernel_gradient = np.dot(gradient, self.rows).reshape(self.kernel.shape)
        bias_gradient = np.sum(gradient, axis=1)

        rows_gradient = np.dot(gradient.T, kernels).reshape(self.output_height, self.output_width, self.channels, self.kernel_height, self.kernel_width)
        input_gradient = np.zeros(self.input.shape)

        for h in range(self.kernel_height):
            for w in range(self.kernel_width):
                input_gradient[:, h:h+self.output_height, w:w+self.output_width] += rows_gradient[:, :, :, h, w].transpose(2, 0, 1)

        self.kernel -= learning_rate * kernel_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient