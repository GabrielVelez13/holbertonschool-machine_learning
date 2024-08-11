#!/usr/bin/env python3
""" 5-log_on_fire Figure with 5 plots """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def all_in_one():
    """ Using gridspec to adjust the position of each subplot"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Creating figure and axis using GridSpec
    fig = plt.figure(figsize=(6.4, 4.8))
    gs = gridspec.GridSpec(3, 2, fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # First plot (basic line chart)
    ax1.set_xlim(0, 10)
    ax1.plot(y0, color="red")

    # Second plot (Scatter)
    # Labels
    ax2.set_title("Men's Height vs Weight", fontsize="x-small")
    ax2.set_xlabel("Height (in)", fontsize="x-small")
    ax2.set_ylabel("Weight (lbs)", fontsize="x-small")

    # Plot
    ax2.scatter(x1, y1, color="magenta")

    # Third Plot (Exponential)
    # Annotations
    ax3.set_title("Exponential Decay of C-14", fontsize="x-small")
    ax3.set_xlabel("Time (years)", fontsize="x-small")
    ax3.set_ylabel("Fraction Remaining", fontsize="x-small")

    # Limits, Scale and plot
    ax3.set_xlim(0, 28650)
    ax3.set_yscale("log")
    ax3.plot(x2, y2)

    # Fourth plot (two lines)
    # Labels
    ax4.set_title("Exponential Decay of Radioactive Elements",
                  fontsize="x-small")
    ax4.set_xlabel("Time (years)", fontsize="x-small")
    ax4.set_ylabel("Fraction Remaining", fontsize="x-small")

    # Limits
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)

    # Plots and legend
    ax4.plot(x3, y31, linestyle="--", label="C-14", color="red")
    ax4.plot(x3, y32, label="Ra-226", color="green")
    ax4.legend()

    # Fifth plot (bar chart)
    # Labels
    ax5.set_title("Project A", fontsize="x-small")
    ax5.set_xlabel("Grades", fontsize="x-small")
    ax5.set_ylabel("Number of Students", fontsize="x-small")

    # Limits
    ax5.set_xlim(0, 100)
    ax5.set_xticks(np.arange(0, 101, 10))
    ax5.set_ylim(0, 30)
    bins = np.arange(0, 110, 10)

    # Plot
    ax5.hist(student_grades, bins=bins, edgecolor="black")

    # Show, label and edit the layout of the plots
    plt.suptitle("All in One")
    plt.tight_layout(h_pad=1, pad=.2, w_pad=1)
    plt.show()
