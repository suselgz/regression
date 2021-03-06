# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:10:23 2020

@author: lgz
"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# example data
mu = 0  # mean of distribution
sigma = 2  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 100
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('')
plt.ylabel('')
plt.title(r'Histogram of IQ: $\E=0, $\D=2$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()