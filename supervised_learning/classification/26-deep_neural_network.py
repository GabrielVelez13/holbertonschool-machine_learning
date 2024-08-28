#!/usr/bin/env python3
""" This module defines a deep neural network class for binary classification"""
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle


class DeepNeuralNetwork:
    """ This class defines a deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """ Initialize the deep neural network """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Return the number of layers """
        return self.__L

    @property
    def cache(self):
        """ Return the cache dictionary """
        return self.__cache

    @property
    def weights(self):
        """ Return the weights dictionary """
        return self.__weights

    def forward_prop(self, X):
        """ Perform forward propagation """
        self.cache['A0'] = X
        for i in range(self.__L):
            W = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            A_prev = self.cache['A' + str(i)]
            Z = np.dot(W, A_prev) + b
            self.cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.cache['A' + str(self.__L)], self.cache

    def cost(self, Y, A):
        """ Compute the cost using logistic regression """
        m = Y.shape[1]
        logprob = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprob)
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the network's predictions """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Perform one pass of gradient descent """
        m = Y.shape[1]
        dZ = cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.dot(self.weights['W' + str(i)].T, dZ) * (A_prev * (1 - A_prev))
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Train the deep neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Save the object to a pickle file """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """ Load the object from a pickle file """
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as file:
            return pickle.load(file)