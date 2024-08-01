#!/usr/bin/env python3
""" concatenate two arrays """


def cat_arrays(arr1, arr2):
    """ Copies the first array and appends the content of the second one """
    new_list = arr1.copy()

    for num in arr2:
        new_list.append(num)

    return new_list
