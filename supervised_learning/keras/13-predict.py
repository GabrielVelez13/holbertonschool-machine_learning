#!/usr/bin/env python3
""" Using a model to predict """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Given some data a model does its prediction """
    prediction = network.predict(data, verbose=verbose)
    return prediction
