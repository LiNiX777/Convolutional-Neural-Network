import numpy as np
from layer import Layer

class Fully_Connected_layer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.random.randn(output_size) * 0.1

    def forward_pass(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backpropagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        bias_gradient = np.sum(output_gradient, axis=0)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient