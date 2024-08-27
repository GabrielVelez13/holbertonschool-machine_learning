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

        # output layer first
        outError = A2 - Y

        outGradient = np.dot(A1, outError.T) / m
        outBias = np.sum(outError, axis=1, keepdims=True) / m

        # hidden layers second
        hidError = np.dot(self.__W2.T, outError) * A1 * (1 - A1)
        hidGradient = np.dot(X, hidError.T) / m
        hidBias = np.sum(hidError, axis=1, keepdims=True) / m

        # Calculate weights and biases
        self.__W1 -= alpha * hidGradient.T
        self.__W2 -= alpha * outGradient.T
        self.__b1 -= alpha * hidBias
        self.__b2 -= alpha * outBias

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train the model """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
