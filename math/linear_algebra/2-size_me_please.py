#!/usr/bin/env python3
""" Function to return the shape of a matrix. """


def matrix_shape(matrix: list):
    """
    Returns the shape of a matrix.
    """
    dimensions = []
    while isinstance(matrix, list):
        dimensions.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return dimensions
