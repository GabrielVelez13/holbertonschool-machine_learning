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
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
