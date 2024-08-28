#!/usr/bin/env python3
""" Starting deep neural networks """
import numpy as np


class DeepNeuralNetwork:
    """ Deep class """
    def __init__(self, nx, layers):
        """
        Validata nx and layers
        init L, cache and weights using he et al. method
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not layers:
            raise TypeError('layers must be a list of positive '
                            'integers')
        if layers[-1] < 1:
            raise TypeError('layers must be a list of positive '
                            'integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = (np.random.randn(layers[0], nx) *
                                      np.sqrt(2 / nx))
            else:
                self.weights[f'W{i+1}'] = (
                        np.random.randn(layers[i], layers[i-1]) *
                        np.sqrt(2 / layers[i-1]))
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function """
        return self.__L

    @property
    def cache(self):
        """ Getter function """
        return self.__cache

    @property
    def weights(self):
        """ Getter function """
        return self.__weights
