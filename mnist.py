import numpy as np
from keras.datasets import mnist

from conv import Convolutional_layer
from dense import Fully_Connected_layer
from activations import ReLU_layer, SoftMax_layer
from reshape import Maxpool_layer, Flatten_layer
from loss import categorical_cross_entropy, categorical_cross_entropy_prime
from train import train, predict, accuracy
from visualization import make_all_plots

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(x, y, limit):
    x, y = x[:limit], y[:limit]

    # normalize x
    x = x.astype(np.float64) / 255
    x = x.reshape(len(x), 1, 28, 28)

    # one hot encode y
    y = np.eye(10)[y.flatten()]
    y = y.reshape(len(y), 1, 10)

    return x, y

np.random.seed(0)
x_train, y_train = preprocess(x_train, y_train, 60000)
x_test, y_test = preprocess(x_test, y_test, 10000)


network = [
    Convolutional_layer((1, 28, 28), (3, 3), 8),
    ReLU_layer(),
    Maxpool_layer((8, 26, 26), (2, 2), 2),    
    Flatten_layer((8, 13, 13), (1, 8 * 13 * 13)),
    Fully_Connected_layer(8 * 13 * 13, 10),   
    SoftMax_layer(),
]

train(network, 
      categorical_cross_entropy, 
      categorical_cross_entropy_prime,
      x_train, 
      y_train, 
      x_test, 
      y_test, 
      epochs=5, 
      learning_rate=0.005
      )

make_all_plots(network, x_test, y_test, folder="results")