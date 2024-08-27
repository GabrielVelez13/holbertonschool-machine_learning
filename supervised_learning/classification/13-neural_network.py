#!/usr/bin/env python3
""" Beginnings of a neural network """
import numpy as np


class NeuralNetwork:
    """ Class where it happens """
    def __init__(self, nx, nodes):
        """
        init the weights, biases, and activations of the hidden
        layer and the output layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((1, nodes)).T
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Returns desired private attribute """
        return self.__W1

    @property
    def b1(self):
        """ Returns desired private attribute """
        return self.__b1

    @property
    def A1(self):
        """ Returns desired private attribute """
        return self.__A1

    @property
    def W2(self):
        """ Returns desired private attribute """
        return self.__W2

    @property
    def b2(self):
        """ Returns desired private attribute """
        return self.__b2

    @property
    def A2(self):
        """ Returns desired private attribute """
        return self.__A2

    @staticmethod
    def sigmoid(d):
        """ Static function for sigmoid"""
        return 1 / (1 + np.exp(-d))

    def forward_prop(self, X):
        """ Forward propagates through hidden and output layer"""
        d1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(d1)
        d2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(d2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost between the true label and out
        output
        """
        return (-np.sum((Y * np.log(A)) + (1 - Y) *
                        np.log(1.0000001 - A)) / Y.shape[1])

    def evaluate(self, X, Y):
        """ Evaluates the neural networks predictions"""
        self.forward_prop(X)
        return np.where(self.__A2 >= .5, 1, 0), self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ gradient descent of a neural network """
        m = Y.shape[1]

        # Calculate the error for the output layer
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Calculate the error for the hidden layer
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update the weights and biases
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NeuralNetwork(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)