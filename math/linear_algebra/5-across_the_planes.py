#!/usr/bin/env python3
"""
Add two matrices
The matrix shape function ensure both matrices are [2 x 2]
"""


def add_matrices2D(mat1, mat2):
    """
    Adding two [2 x 2] matrices
    With a modification to the if-statement
    it could work with any matrices of the same shape
    as long as they are square
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    added_matrix = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        added_matrix.append(row)
    return added_matrix
