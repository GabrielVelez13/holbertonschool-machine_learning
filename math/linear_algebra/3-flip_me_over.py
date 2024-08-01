#!/usr/bin/env python3
# Transpose a matrix


def matrix_transpose(matrix: list) -> list:
    """
    Used to transpose the matrix.
    :param matrix: must be a list.
    :return: Returns a transpose list of the matrix
    """
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transposed
