#!/usr/bin/env python3
""" Calculating accuracy """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    :param y: Labels
    :param y_pred: Predictions by network
    :return: Returns the accuracy of the predictions
    """

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
