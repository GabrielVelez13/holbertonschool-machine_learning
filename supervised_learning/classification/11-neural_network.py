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
