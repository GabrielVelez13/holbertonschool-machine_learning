#!/usr/bin/env python3
""" Starting deep neural networks """
import numpy as np
import pickle


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
                self.weights[f'W{i + 1}'] = (
                        np.random.randn(layers[i], layers[i - 1]) *
                        np.sqrt(2 / layers[i - 1]))
            self.weights[f'b{i + 1}'] = np.zeros((layers[i], 1))

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

    @staticmethod
    def sigmoid(X):
        """ Returns the sigmoid function """
        return 1 / (1 + np.exp(-X))

    def forward_prop(self, X):
        """ Forward propagating the deep network """

        self.__cache['A0'] = X
        for i in range(self.L):
            W = self.weights[f'W{i + 1}']
            b = self.weights[f'b{i + 1}']
            A_prev = self.cache[f'A{i}']

            z = np.dot(W, A_prev) + b
            self.__cache[f'A{i + 1}'] = self.sigmoid(z)
        return self.__cache[f'A{self.L}'], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost function using logistic regression """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 -
                                                         A))) / m

    def evaluate(self, X, Y):
        """ Evaluates outputs prediction and cost """
        self.forward_prop(X)
        cost = self.cost(Y, self.cache[f'A{self.L}'])
        return np.where(self.cache[f'A{self.L}'] >= .5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates the gradient descent """
        m = Y.shape[1]
        dZ = cache[f'A{self.L}'] - Y
        for i in range(self.L, 0, -1):
            A_prev = cache[f'A{i - 1}']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ_step1 = np.dot(self.weights[f'W{i}'].T, dZ)
            dZ = dZ_step1 * (A_prev * (1 - A_prev))
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Train the neuron """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 1:
            raise ValueError('alpha must be positive')
        if not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step < 1 or step > iterations:
            raise ValueError('step must be a positive integer and '
                             '<= iterations')

        costs = []

        for i in range(iterations + 1):
            cache = self.forward_prop(X)[1]
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == iterations or i == 0:
                cost = self.cost(Y, self.cache[f'A{self.L}'])
                costs.append((cost, i))
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            costs.append(((self.cost(Y, self.cache[f'A{self.L}'])),
                         iterations))
            y_values, x_values = zip(*costs)
            plt.plot(x_values, y_values, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
