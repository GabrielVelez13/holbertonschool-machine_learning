#!/usr/bin/env python3
""" Function to return the shape of a matrix. """
import numpy as np


def matrix_shape(matrix: list) -> list:
    """
    Returns the shape of a matrix by
    turning it into a numpy array
    """
    return list(np.array(matrix).shape)
