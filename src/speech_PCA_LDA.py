##
# @file   voice_pca_lda.py 
# @brief  An attempt for voice recognition classifier using pca, lda
# @author Marek Sarvas

from ikrlib import *

import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
import sys
from scipy.ndimage import gaussian_filter

# paths to data directories
TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'
THRESHOLD = 100000 # the evaluation treshold for the test data score

# load target and non target voice data
train_t = list(wav16khz2mfcc(TRAIN_TARGET).values())
train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values())


# convert to numpy arrays
train_t = np.vstack(train_t)
train_n = np.vstack(train_n)
dim = train_t.shape[1]

# standardise the data for PCA, in LDA this makes no difference
train_mean = np.mean(np.vstack((train_t, train_n)), axis=0) 
train_std = np.std(np.vstack((train_t, train_n)), axis=0)
train_t -= train_mean
train_t /= train_std
train_n -= train_mean
train_n /= train_std

cov_tot = np.cov(np.vstack([train_t, train_n]).T, bias=True)

# PCA
d_pca, e_pca = scipy.linalg.eigh(cov_tot, eigvals=(dim-1, dim-1))
# one dimensional data
x1_pca = train_t.dot(e_pca)
x2_pca = train_n.dot(e_pca)

# plot the result, not great
plt.figure()
junk = plt.hist(x1_pca, 40, histtype='step', color='b')
junk = plt.hist(x2_pca, 40, histtype='step', color='r')
plt.show()

#LDA
n_x1 = len(train_t)
n_x2 = len(train_n)

# within and across class, cov_tot = ac+wc
cov_wc = (n_x1*np.cov(train_t.T, bias=True) + n_x2*np.cov(train_n.T, bias=True)) / (n_x1 + n_x2)
cov_ac = cov_tot - cov_wc
d_lda, e_lda = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1)) 

# one dimensional data
x1_lda = train_t.dot(e_lda)
x2_lda = train_n.dot(e_lda)

# plot the result, also not great
plt.figure()
junk = plt.hist(x1_lda, 40, histtype='step', color='b')
junk = plt.hist(x2_lda, 40, histtype='step', color='r')
plt.show()
