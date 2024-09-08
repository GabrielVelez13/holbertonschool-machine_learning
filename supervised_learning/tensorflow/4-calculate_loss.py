#!/usr/bin/env python3
""" Calculates the softmax cross entropy lost """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    :param y: labels
    :param y_pred: predictions of the network
    :return: the reduced mean of a softmax cross entropy lost
    """
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred),
        name='softmax_cross_entropy_loss/value')
