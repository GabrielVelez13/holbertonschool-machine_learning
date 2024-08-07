#!/usr/bin/env python3
""" Plot exponential decay """
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """ Plot a line where the y-axis on logarithmic scale """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Annotations
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Limits and Scale
    plt.xlim(0, 28650)
    plt.yscale("log")

    # Showcasing
    plt.plot(x, y)
    plt.show()
