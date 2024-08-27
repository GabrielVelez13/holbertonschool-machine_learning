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
        if list(filter(lambda n: n >= 1, layers)):
            raise TypeError('layers must be a list of positive '
                            'integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))
