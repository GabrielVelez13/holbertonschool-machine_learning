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
