#!/usr/bin/env python3
# Transpose a matrix
import numpy as np


def matrix_transpose(matrix: list) -> list:
    """
    Uses numpy to transpose the matrix.
    :param matrix: must be a list.
    :return: Returns a transpose list of the matrix
    """
    return np.array(matrix).transpose().tolist()
