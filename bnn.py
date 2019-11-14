# Binary Neural Network

import numpy as np
import matplotlib.pyplot as plt
import math
from activations import sigmoid, tanh, relu, softmax, sign

# log loss
from sklearn.metrics import log_loss

def nll(y_true, y_pred):
    """ Negative log-likelihood.
    """
    loss = 0
    for i in range(len(y_true)):
        loss -= np.log(y_pred[i, y_true[i]])
    loss = -np.sum(np.log(y_pred[range(len(y_true)), y_true])) / len(y_true)
    return loss

class BNN():

    def __init__(self, layers=[10, 10, 10], W=None, B=None):
        """ Feed-forward Binary Neural Network.

            Initialize the neural network weights, activation function and return the number of parameters

            :param layers: The number of units per layer - list of int
            :param W: List of weight matrices, None for random initialization.
            :param B: List of bias matrices, None for random initialization.
        """
        self.layers = layers # architecture
        self.act_func = sign  # Use sign activation function
        if W is None:
            W = []
            for i in range(np.size(layers) - 1):
                # weights initialization
                w = np.random.choice([-1, 1], size=(layers[i + 1], layers[i])) # binary weights
                W.append(w.astype(np.int8))
        if B is None:
            B = []
            for i in range(np.size(layers) - 1):
                # bias initialization
                b = np.random.choice([-1, 1], size=(W[i].shape[0], 1)) # binary bias
                B.append(b.astype(np.int8))
        self.W = W
        self.B = B

    def predict_proba(self, X):
        """ Perform the forward propagation.

            :param X: The batch - np.ndarray
            :return: A list of activation values - list of np.ndarray
        """
        X = X.T
        for i in range(len(self.W) - 1):
            X = self.act_func(np.dot(self.W[i], X) + self.B[i])
        X = softmax(np.dot(self.W[-1], X) + self.B[-1]) # softmax on last layer
        return X.T

    def predict(self, X):
        """ Perform the forward propagation and output the argmax target.

            :param X: Batch of data - np.ndarray
            :return: Predicted target index
        """
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1)

    def loss(self, X, y_true, loss_function=None):
        """ Compute the loss value of the current network on the full batch

            :param X: Batch of data - np.ndarray
            :param y_true: Labels corresponding to the batch - np.ndarray
            :param loss_function: None for logloss or 'nll'
            :return: Loss - float
        """
        y_pred = self.predict_proba(X)
        if loss_function is None:
            loss = log_loss(y_true, y_pred)
        elif loss_function == 'nll':
            loss = nll(y_true, y_pred)
        else:
            raise Exception('Unknown loss function: {}'.format(loss_function))
        return loss

if __name__ == "__main__":
    print('testing...')

    print('log loss')
    y_true = np.array([[1,0,0,0],
                   [0,0,0,1]])
    y_pred = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.97]])
    print(log_loss(y_true, y_pred))
    print(_log_loss(y_true, y_pred))

    model = BNN(layers=[2, 4])
    X = np.array([[1, 3], [0, -3]])
    y = np.array([0, 1])
    print('layers')
    print(model.layers)
    print('bias')
    print(model.B)
    print('weights')
    print(model.W)
    print('input')
    print(X)
    print('labels')
    print(y)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    print('loss')
    print(model.loss(X, y))
