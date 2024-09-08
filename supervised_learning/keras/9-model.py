#!/usr/bin/env python3
""" Save and load a model using Keras """
import tensorflow.keras as K


def save_model(network, filename):
    """ Save a model """
    network.save(filename)


def load_model(filename):
    """ Load a model """
    return K.models.load_model(filename)
