#!/usr/bin/env python3
""" Find the derivative of a polynomial """


def poly_derivative(poly):
    """ List comprehension and slicing to return the derivative """
    if not isinstance(poly, list):
        return None

    if len(poly) == 1 or len(poly) == 0:
        return None

    return [i * num for i, num in enumerate(poly)][1:]
