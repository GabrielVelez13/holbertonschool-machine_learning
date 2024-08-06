#!/usr/bin/env python3
""" Making plotting a line """
import numpy as np
import matplotlib.pyplot as plt

def line():
    """ Plotting a like whose values are 1 - 10 cubed """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y)
    plt.show()
