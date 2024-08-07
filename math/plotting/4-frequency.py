#!/usr/bin/env python3
""" Creating a bar chart """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ Plot a bar chart with bins of every 10 units of x"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Labels
    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    # Limits
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0, 30)
    bins = np.arange(0, 110, 10)

    # Plot
    plt.hist(student_grades, bins=bins, edgecolor="black")
    plt.show()
