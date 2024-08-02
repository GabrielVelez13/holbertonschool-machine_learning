#!/usr/bin/env python3
""" Multiply matrices """


def mat_mul(mat1, mat2):
    """
    Checks matrices and multiplies them if able.
    Zip is an MVP
    """
    if len(mat1[0]) != len(mat2):
        return None

    mult_matrix = []
    for row_1 in mat1:
        row = []
        for col_2 in zip(*mat2):
            row.append(sum(a * b for a, b in zip(row_1, col_2)))
        mult_matrix.append(row)
    return mult_matrix
