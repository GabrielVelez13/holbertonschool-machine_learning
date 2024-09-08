#!/usr/bin/env python3
""" Convert to one hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Convert labels into one hot matrix  """
    one_hot_matrix = \
        K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix
