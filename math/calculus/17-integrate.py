#!/usr/bin/env python3
""" Finding the integral of a polynomial """


def poly_integral(poly, C=0):
    """
    dividing each number in the array by its index + 1
    the opposite of a derivatice, then adding C to the
    0th index
    """
    integral = [num / (i + 1) for i, num in enumerate(poly)]
    integral.insert(0, C)
    return integral
