#!/usr/bin/env python3
""" Save and load weights using Keras """
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ Save the weights of a model """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ load the weights of a model """
    network.load_weights(filename)
