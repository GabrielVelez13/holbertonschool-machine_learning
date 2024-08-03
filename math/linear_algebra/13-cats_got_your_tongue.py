#!/usr/bin/env python3
""" concatenate two matrices  """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Using numpy's concatenate to do its
    name same taking into account the axis
    """
    return np.concatenate((mat1, mat2), axis=axis)
