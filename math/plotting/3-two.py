#!/usr/bin/env python3
""" Plot two lines in one graph """
import numpy as np
import matplotlib.pyplot as plt


def two():
    """ Show two plots with the labels of the lines as legend """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Labels
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Limits
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Plots
    plt.plot(x, y1, linestyle="--", label="C-14", color="red")
    plt.plot(x, y2, label="Ra-226", color="green")

    # Show plots with legend
    plt.legend()
    plt.show()
