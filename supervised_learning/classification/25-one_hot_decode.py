#!/usr/bin/env python3
""" hot ones """
import numpy as np


def one_hot_decode(one_hot):
    """
    creating a metrix where the activated value of each column
    is given by a class list
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    try:
        return one_hot.argmax(axis=0)
    except Exception:
        return None

