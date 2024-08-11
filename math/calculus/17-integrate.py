#!/usr/bin/env python3
""" Finding the integral of a polynomial """


def poly_integral(poly, C=0):
    """
    dividing each number in the array by its index + 1
    the opposite of a derivatice, then adding C to the
    0th index
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    integral = []
    for i, num in enumerate(poly):
        coe = num / (i + 1)
        if coe % 1 == 0:
            integral.append(int(coe))
        else:
            integral.append(coe)

    integral.insert(0, C)
    return integral
