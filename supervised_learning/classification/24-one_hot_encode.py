#!/usr/bin/env python3
""" hot ones """
import numpy as np


def one_hot_encode(Y, classes):
    """
    creating a metrix where the activated value of each column
    is given by a class list
    """
    if (not isinstance(Y, np.ndarray) or not isinstance(classes, int)
            or classes <= 0):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
