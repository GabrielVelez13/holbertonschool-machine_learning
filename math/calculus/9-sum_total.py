#!/usr/bin/env python3
""" Do a summation without using any loops """


def summation_i_squared(n):
    """
    Due to not being able to use loops
    recursion was used instead
    """
    if n < 1:
        return None
    if n == 1:
        return 1

    return n**2 + summation_i_squared(n - 1)
