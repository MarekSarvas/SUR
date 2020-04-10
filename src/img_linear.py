#!/usr/bin/python

from ikrlib import *
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
from sys import exit

# paths to training data directories
TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'

# first, load target and non-target training data and convert each image to a 1d array
train_target = png2fea(TRAIN_TARGET) # target training data
train_ntarget = png2fea(TRAIN_NTARGET) # non-target training data

# convert the grayscale images to 1d arrays
x1 = []
x2 = []
for im in train_target.values(): x1.append(im.flatten())
for im in train_ntarget.values(): x2.append(im.flatten())

# convert data to numpy arrays
x1 = np.array(x1) # target
x2 = np.array(x2) # non-target
dim = x1.shape[1]
cov_tot = np.cov(np.vstack([x1, x2]).T, bias=True)

#PCA
"""
# take just 2 largest eigenvalues and corresponding eigenvectors
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))
print(d)

x1_pca = x1.dot(e)
x2_pca = x2.dot(e)
plt.plot(x1_pca[:,1], x1_pca[:,0], 'b.', ms=1)
plt.plot(x2_pca[:,1], x2_pca[:,0], 'r.', ms=1)
plt.show()
"""

#LDA
"""
n_x1 = len(x1)
n_x2 = len(x2)
cov_wc = (n_x1*np.cov(x1.T, bias=True) + n_x2*np.cov(x2.T, bias=True)) / (n_x1 + n_x2)
cov_ac = cov_tot - cov_wc
d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))
plt.figure()
junk = plt.hist(x1.dot(e), 40, histtype='step', color='b', normed=True)
junk = plt.hist(x2.dot(e), 40, histtype='step', color='r', normed=True)
plt.show()
"""


"""
# apriori probability NOTE: 0.5????
pCx1 = len(x1)/(len(x1) + len(x2))
pCx2 = 1 - pCx1

# compute the common covariance matrix
x1_cov = np.cov(x1.T, bias=True)
x2_cov = np.cov(x2.T, bias=True)
"""

