import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import scipy.optimize as opt
from scipy.optimize import curve_fit


def expon_fit(x, a, b):
    return a * math.exp(-b * x)


def power_fit(x, m, c, c0):
    return c0 + (x**m) * c


def lognorm_fit(x, mu, sigma):
    return (1 / (x * sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-(math.log(x, math.e) - mu)**2 / (2 * sigma**2))


if __name__ == '__main__':
    x = []
    y = []
    for i in np.arange(0.01, 5.0, 0.01):
        x.append(i)
        y.append(lognorm_fit(i, 0, 1))
    plt.plot(x, y)
    plt.show()
