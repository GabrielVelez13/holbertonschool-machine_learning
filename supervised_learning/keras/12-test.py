#!/usr/bin/env python3
""" Testing testing testing"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Test a model """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]