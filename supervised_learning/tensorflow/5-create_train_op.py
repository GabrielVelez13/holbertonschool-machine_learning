#!/usr/bin/env python3
""" gradient descent optimizer """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Using gradient descent optimizer to minimize loss
    :param loss: the loss from the network
    :param alpha: the learning rate
    :return: the training of a network using gradient descent
    """
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return opt.minimize(loss)
