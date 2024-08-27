#!/usr/bin/env python3
""" The start of a neuron """
import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step <= 0 or step > iterations:
            raise ValueError('step must be a positive integer and <= '
                             'iterations')


        costs = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step  == 0 or i == iterations or i == 0:
                cost = self.cost(Y, self.__A)
                costs.append((i, cost))
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            costs.append((self.cost(Y, self.__A), iterations))
            y_values, x_values = zip(*costs)
            plt.plot(x_values, y_values, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

lib_train = np.load('data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()