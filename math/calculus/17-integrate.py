#!/usr/bin/env python3
""" Finding the integral of a polynomial """


def poly_integral(poly, C=0):
    """
    dividing each number in the array by its index + 1
    the opposite of a derivatice, then adding C to the
    0th index
    """
    if not isinstance(poly, list) or C is None:
        return None

    if len(poly) == 0:
        return None

    if poly == [0]:
        return [C]

    integral = []
    for i, num in enumerate(poly):
        coe = num / (i + 1)
        if coe % 1 == 0:
            integral.append(int(coe))
        else:
            integral.append(coe)

    integral.insert(0, C)
    return integral
