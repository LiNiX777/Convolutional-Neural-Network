import numpy as np


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward_pass(output)
    return output

def accuracy(network, x_data, y_data):
    correct = 0
    for x, y in zip(x_data, y_data):
        correct += np.argmax(predict(network, x)) == np.argmax(y)
    return correct / len(x_data)

def train(network, loss, loss_prime, x_train, y_train, x_test, y_test, epochs, learning_rate):
    for epoch in range(epochs):
        error = 0
        for i in np.random.permutation(len(x_train)):
            x = x_train[i]
            y = y_train[i].reshape(1, -1)                  

            output = predict(network, x)
            error += loss(y, output)

            gradient = loss_prime(y, output)
            for layer in reversed(network):                  
                gradient = layer.backpropagation(gradient, learning_rate)

        error /= len(x_train)
        acc = accuracy(network, x_test, y_test)
        print(f"epoch {epoch + 1}/{epochs}  loss {error:.4f}  test accuracy {acc:.3f}")

