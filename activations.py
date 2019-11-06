########################
# Fonctions d'activation
########################
import numpy as np

def sign(z):
    return np.sign(z)

def sigmoid(z):
    """
    Perform the sigmoid transformation to the pre-activation values
    Inputs: z: the pre-activation values - ndarray
    Outputs: y: the activation values - ndarray
           : yp: the derivatives w.r.t. z - ndarray
    """
    return (1 / (1 + np.exp(-z)))

def tanh(z):
    """
    Perform the tanh transformation to the pre-activation values
    Inputs: z: the pre-activation values - ndarray
    Outputs: y: the activation values - ndarray
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu(z):
    """
    Perform the relu transformation to the pre-activation values
    Inputs: z: the pre-activation values - ndarray
    Outputs: y: the activation values - ndarray
    """
    return np.maximum(0, z)
    
def softmax(z):
    """
    Perform the softmax transformation to the pre-activation values
    Inputs: z: the pre-activation values - ndarray
    Outputs: out: the activation values - ndarray
    """
    return np.exp(z - np.max(z, 0)) / np.sum(np.exp(z - np.max(z, 0)), axis=0)
