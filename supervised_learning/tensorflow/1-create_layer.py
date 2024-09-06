#!/usr/bin/env python3
""" Creating a custom layer """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    creating a layer that uses He et. al initialization of
    weights and in a Dense layer with n nodes given the weights and
    activation
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(name='layer', kernel_initializer=init,
                            activation=activation, units=n)
    return layer(prev)
