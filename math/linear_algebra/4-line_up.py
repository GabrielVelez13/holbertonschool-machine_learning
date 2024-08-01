#!/usr/bin/env python3
"""
Adding arrays and checking if they are equal in length.
"""


def add_arrays(arr1, arr2):
    """
    Using list comprehension to sum both arrays
    :param arr1: First array to be added
    :param arr2: Second array to be added
    :return: The sum of the two arrays in a new array
    """
    if len(arr1) != len(arr2):
        return None
    added_arr = [arr1[i] + arr2[i] for i in range(len(arr1))]
    return added_arr
