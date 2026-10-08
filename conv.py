import numpy as np
from layer import Layer

class Convolutional_layer(Layer):
    def __init__(self, input_shape, kernel_shape, kernel_num):
        self.channels, self.height, self.width = input_shape
        self.kernel_height, self.kernel_width = kernel_shape
        self.kernel_num = kernel_num
        self.output_height = self.height - self.kernel_height + 1
        self.output_width = self.width - self.kernel_width + 1
        self.kernel = np.random.randn(self.kernel_num, self.channels, self.kernel_height, self.kernel_width) * 0.1
        self.bias = np.random.randn(self.kernel_num) * 0.1
        
    def forward_pass(self, input):
        self.input = input
        output = np.zeros((self.kernel_num, self.output_height, self.output_width))

        for i in range(self.kernel_num):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    output[i,h,w] = np.sum(self.input[:,h:h+self.kernel_height, w:w+self.kernel_width] * self.kernel[i]) + self.bias[i]
        return output
    
    def backpropagation(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernel.shape)
        input_gradient = np.zeros(self.input.shape)
        bias_gradient = np.sum(output_gradient, axis=(1, 2))

        for i in range(self.kernel_num):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    window = self.input[:, h:h+self.kernel_height, w:w+self.kernel_width]
                    kernel_gradient[i] += output_gradient[i, h, w] * window

                    input_gradient[:, h:h+self.kernel_height, w:w+self.kernel_width] += output_gradient[i, h, w] * self.kernel[i]

        self.kernel -= learning_rate * kernel_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient




