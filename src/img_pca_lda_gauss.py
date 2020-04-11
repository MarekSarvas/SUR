#!/usr/bin/python3.7

##
# @file   img_pca_lda_gauss.py 
# @brief  A shaky face recognition classifier. Utilizes pca, lda, and a gaussian
#         probability distribution.
# @author Simon Sedlacek

from ikrlib import *
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
from sys import exit
from scipy.ndimage import gaussian_filter

# paths to data directories
TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'
THRESHOLD = 900000 # the evaluation treshold for the test data score

# first, load target and non-target training and test data and convert each image
# to a 1d array
train_target = png2fea(TRAIN_TARGET) # target training data
train_ntarget = png2fea(TRAIN_NTARGET) # non-target training data
test_target = list(png2fea(TEST_TARGET).items()) # target test data
test_ntarget = list(png2fea(TEST_NTARGET).items()) # non-target test data

# convert the grayscale images to 1d arrays
x1 = []
x2 = []
#image = gaussian_filter(list(train_target.values())[0], 1)
#imageio.imwrite('hello.png', image)
for im in train_target.values(): x1.append(gaussian_filter(im, 2).flatten())
for im in train_ntarget.values(): x2.append(gaussian_filter(im, 2).flatten())
#for im in train_target.values(): x1.append(im.flatten())
#for im in train_ntarget.values(): x2.append(im.flatten())

# convert data to numpy arrays
x1 = np.array(x1) # target
x2 = np.array(x2) # non-target
dim = x1.shape[1]

# standardise the data
train_mean = np.mean(np.vstack((x1, x2)), axis=0) 
train_std = np.std(np.vstack((x1, x2)), axis=0)
x1 -= train_mean
x1 /= train_std
x2 -= train_mean
x2 /= train_std
cov_tot = np.cov(np.vstack([x1, x2]).T, bias=True)

# PCA - reduce the dimensionality to 150 dimensions, otherwise the LDA won't work...
d_pca, e_pca = scipy.linalg.eigh(cov_tot, eigvals=(dim-150, dim-1))
x1_pca = x1.dot(e_pca)
x2_pca = x2.dot(e_pca)

# plot the pca reslut, nothing interesting...
#plt.plot(x1_pca[:,1], x1_pca[:,0], 'b.', ms=1)
#plt.plot(x2_pca[:,1], x2_pca[:,0], 'r.', ms=1)
#plt.show()

# perform the LDA on the reduced data space
cov_tot_pca = np.cov(np.vstack([x1_pca, x2_pca]).T, bias=True)
dim_pca = x1_pca.shape[1]

n_x1 = len(x1_pca)
n_x2 = len(x2_pca)
cov_wc = (n_x1*np.cov(x1_pca.T, bias=True) + n_x2*np.cov(x2_pca.T, bias=True)) / (n_x1 + n_x2)
cov_ac = cov_tot_pca - cov_wc
d_lda, e_lda = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim_pca-1, dim_pca-1))

# now we've got our one dimensional data
x1_lda = x1_pca.dot(e_lda)
x2_lda = x2_pca.dot(e_lda)

# ... and plot the lda result.. beautiful...
plt.figure()
junk = plt.hist(x1_lda, 40, histtype='step', color='b')
junk = plt.hist(x2_lda, 40, histtype='step', color='r')
plt.show()

# compute the gaussian distributions for our classes and evalueate the test data
apriori = 0.5
mean_x1, cov_x1 = train_gauss(x1_lda)
mean_x2, cov_x2 = train_gauss(x2_lda)

# plot the gaussians... awesome..
#plt.figure()
#gauss_x1 = np.linspace(mean_x1 - 10*cov_x1, mean_x1 + 10*cov_x1, 100)
#gauss_x2 = np.linspace(mean_x2 - 10*cov_x2, mean_x2 + 10*cov_x2, 100)
#plt.plot(gauss_x1, stats.norm.pdf(gauss_x1, mean_x1, cov_x1))
#plt.plot(gauss_x2, stats.norm.pdf(gauss_x2, mean_x2, cov_x2))
#plt.show()

total = 0
ok = 0
print('======Target test data evaluation======')
for filename, data in test_target:
    total += 1
    # standardise the data
    data = data.flatten()
    data -= train_mean
    data /= train_std
    
    data = (data.dot(e_pca)).dot(e_lda) # transform the test data
    ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
    ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))

    # indicates whether the data point was correctly classified
    if int(sum(ll_target) - sum(ll_ntarget)) > THRESHOLD:
        ok += 1
        correct = True
    else: correct = False
    
    print(correct, int(sum(ll_target) - sum(ll_ntarget)), filename)

print((ok/total) * 100)

print('')
print('======Non-target test data evaluation======')
total = 0
ok = 0
for filename, data in test_ntarget:
    total += 1
    # standardise the data
    data = data.flatten()
    data -= train_mean
    data /= train_std
    
    data = (data.dot(e_pca)).dot(e_lda)
    ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
    ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))
    
    # indicates whether the data point was correctly classified
    if int(sum(ll_target) - sum(ll_ntarget)) < THRESHOLD:
        ok += 1
        correct = True
    else: correct = False
          
    print(correct, int(sum(ll_target) - sum(ll_ntarget)), filename)

print((ok/total) * 100)
