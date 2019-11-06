# Binary Neural Network

import numpy as np
import matplotlib.pyplot as plt
import math
from activations import sigmoid, tanh, relu, softmax, sign

class BNN():

    def __init__(self, layers=[10, 10, 10], act_func_name='sign', W=None, B=None):
        """
        Feed-forward Binary Neural Network.
        Initialize the neural network weights, activation function and return the number of parameters
        Inputs: layers: the number of units per layer -  list of int
              : act_func_name: the activation function name (sigmoid, tanh or relu) - str
        Outputs: W: a list of weights for each hidden layer - list of ndarray
               : B: a list of bias for each hidden layer - list of ndarray
               : act_func: the activation function - function
        """
        self.layers = layers # architecture
        self.act_func = globals()[act_func_name]  # Cast the string to a function
        if W is None:
            W = []
            for i in range(np.size(layers) - 1):
                # weights initialization
                w = np.random.choice([-1, 1], size=(layers[i + 1], layers[i])) # binary weights
                W.append(w)
        if B is None:
            B = []
            for i in range(np.size(layers) - 1):
                # bias initialization
                b = np.zeros((W[i].shape[0], 1))
                B.append(b)
        self.W = W
        self.B = B

    def predict_proba(self, X):
        """
        Perform the forward propagation
        Inputs: X: the batch - ndarray
        Outputs: a list of activation values - list of ndarray
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = X.T
        for i in range(len(self.W) - 1):
            X = self.act_func(np.dot(self.W[i], X))# + self.B[i])
            #print(X)
        X = softmax(np.dot(self.W[-1], X)) # + self.B[-1] # softmax on last layer
        return X.T

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1)

    def loss(self, X, y_true):
        """
        Compute the loss value of the current network on the full batch
        Inputs: X: the batch - ndarray
              : labels: the labels corresponding to the batch
        Outputs: loss: the negative log-likelihood - float
               : accuracy: the ratio of examples that are well-classified - float
        """
        y_pred = self.predict_proba(X)
        loss = 0
        for i in range(len(X)):
            loss -= np.log(y_pred[y_true[i], i])
        loss = -np.sum(np.log(y_pred[y_true, range(len(y_true))])) / len(y_true)
        return loss

if __name__ == "__main__":
    model = BNN(layers=[2, 4])
    X = [[1, 3], [0, -3]]
    y = [0, 1]
    print('layers')
    print(model.layers)
    print('bias')
    print(model.B)
    print('weights')
    print(model.W)
    print('input')
    print(X)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    print('labels')
    print(y)
    print('loss')
    print(model.loss(X, y))
