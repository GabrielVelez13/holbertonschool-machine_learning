#!/usr/bin/env python3
""" Do a summation without using any loops """


def summation_i_squared(n):
    """
    As this is just a sum of squares we use the formula
    """
    if n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
