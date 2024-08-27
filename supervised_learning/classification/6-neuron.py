#!/usr/bin/env python3
""" The start of a neuron """
import numpy as np


class Neuron:
    """ A Neuron class where it all happens """

    def __init__(self, nx):
        """
        Initializing checking the validity of nx
        and creating the weight vector, bias, and activation
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be a integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)

        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 -
                                                        A)) / m

    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(A >= .5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        error = A - Y
        weightGradient = np.dot(X, error.T) / m

        biasGradient = np.sum(error) / m

        self.__W -= alpha * weightGradient.T
        self.__b -= alpha * biasGradient

    def train(self, X, Y, iterations=5000, alpha=0.05):
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
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
