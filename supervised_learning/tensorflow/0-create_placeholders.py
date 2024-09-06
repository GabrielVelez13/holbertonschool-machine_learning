#!/usr/bin/env python3
""" Using placeholders in TensorFlow """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ placeholder """
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, nx),
                                 name='x')
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, classes),
                                 name='y')
    return x, y