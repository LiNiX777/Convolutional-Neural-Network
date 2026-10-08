import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-12))

def categorical_cross_entropy_prime(y_true, y_pred):
    return -y_true / (y_pred + 1e-12)