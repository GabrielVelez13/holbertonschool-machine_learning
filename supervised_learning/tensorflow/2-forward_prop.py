#!/usr/bin/env python3
""" Forward propagating through layers """
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_size=[], activation=[]):
    """ going throughout a certain number of layers
    and activations to forward prop """
    prev = x
    for size, activation in zip(layer_size, activation):
        pred = create_layer(prev, size, activation)
    return pred
