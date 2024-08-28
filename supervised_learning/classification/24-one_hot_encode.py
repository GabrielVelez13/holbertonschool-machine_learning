#!/usr/bin/env python3
"""  """
import numpy as np


def one_hot_encode(Y, classes):
    """
    creating a metrix where the activated value of each column
    is given by a class list
    """
    z = np.zeros((Y.shape[0], classes ))
    z[Y, np.arange(classes)] = 1
    return z
