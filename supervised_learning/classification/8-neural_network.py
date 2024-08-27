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

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((1, nodes)).T
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
