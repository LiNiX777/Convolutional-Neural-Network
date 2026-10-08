import numpy as np
from layer import Layer

class ReLU_layer(Layer):   
    def forward_pass(self, input):
        self.input = input
        return np.maximum(0, input)

    def backpropagation(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)


class SoftMax_layer(Layer):
    def forward_pass(self, input):
        e = np.exp(input - np.max(input))
        self.output = e / np.sum(e)
        return self.output

    def backpropagation(self, output_gradient, learning_rate):
        n = self.output
        return n * (output_gradient - np.sum(output_gradient * n))