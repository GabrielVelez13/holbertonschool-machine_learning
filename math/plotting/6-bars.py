#!/usr/bin/env python3
""" Creating a stacked bar plot """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Unpacked data into tuples with
    corresponding color and plotting them"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Unpacking and organizing data
    x = ["Farrah", "Fred", "Felicia"]
    y1 = (fruit[0], "red")
    y2 = (fruit[1], "yellow")
    y3 = (fruit[2], "#ff8000")
    y4 = (fruit[3], "#ffe5b4")

    # Plotting
    plt.bar(x, y1[0], color=y1[1],
            label="apple", width=.5)
    plt.bar(x, y2[0], color=y2[1], bottom=y1[0],
            label="banana", width=.5)
    plt.bar(x, y3[0], color=y3[1], bottom=y1[0] + y2[0],
            label="oranges", width=.5)
    plt.bar(x, y4[0], color=y4[1], bottom=y1[0] + y2[0] + y3[0],
            label="peaches", width=.5)

    # Labels, Limit and Legend
    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.legend()

    plt.show()
