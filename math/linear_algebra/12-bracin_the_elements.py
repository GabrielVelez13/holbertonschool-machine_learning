#!/usr/bin/env python3
""" Perform element wise operation on two matrices """


def np_elementwise(mat1, mat2):
    """ Returns a tuple of the various operations """
    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)
