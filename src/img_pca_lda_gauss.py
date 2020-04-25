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
from scipy.ndimage import rotate

# paths to data directories
TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'
EVAL = '../../SUR_projekt2019-2020_eval/eval/'
THRESHOLD = 0.0 # the evaluation treshold for the test data score

# first, load target and non-target training and test data
train_target = png2fea(TRAIN_TARGET) # target training data
train_ntarget = png2fea(TRAIN_NTARGET) # non-target training data
test_target = list(png2fea(TEST_TARGET).items()) # target test data
test_ntarget = list(png2fea(TEST_NTARGET).items()) # non-target test data
eval_data = list(png2fea(EVAL).items()) # evaluation data

def crop_image(im, st1, end1, st2, end2):
    end1 = im.shape[0] - end1
    end2 = im.shape[1] - end2
    return im[st1+11:end1-11, st2+11:end2-11]

def zoom_crop(im, zoom):
    len0 = im.shape[0]
    im = scipy.ndimage.zoom(im, zoom)
    len1 = im.shape[0]
    diff = len1 - len0
    # crop the image 
    return im[diff//2:len1-diff//2, diff//2:len1-diff//2]

def brighten_darken(im):
    im1 = np.clip(im + 30, 0, 255)
    im2 = np.clip(im - 30, 0, 255)
    return im, im1, im2

def inflate_data(im):
    data = []
    for image in brighten_darken(im):
        rot1 = rotate(image, 10, mode='nearest', reshape=False);
        rot2 = rotate(image, -10, mode='nearest', reshape=False);
        rot3 = rotate(image, 5, mode='nearest', reshape=False)
        rot4 = rotate(image, -5, mode='nearest', reshape=False)
        for im in [image, rot1, rot2, rot3, rot4]:
            for im2 in [im, zoom_crop(im, 1.1)]:
                data.append(crop_image(im, 2, 2, 2, 2).flatten())
                data.append(crop_image(im, 2, 2, 4, 0).flatten())
                data.append(crop_image(im, 2, 2, 0, 4).flatten())
                data.append(crop_image(im, 0, 4, 0, 4).flatten())
                data.append(crop_image(im, 0, 4, 2, 2).flatten())
                data.append(crop_image(im, 0, 4, 4, 0).flatten())
                data.append(crop_image(im, 4, 0, 0, 4).flatten())
                data.append(crop_image(im, 4, 0, 2, 2).flatten())
                data.append(crop_image(im, 4, 0, 4, 0).flatten())

    return data

# convert the grayscale images to 1d arrays
x1 = []; x2 = []
for im in train_target.values():
    x1.append(inflate_data(im))
for im in train_ntarget.values():
    x2.append(inflate_data(im))
x1 = np.vstack(x1)
x2 = np.vstack(x2)

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

# PCA - reduce the dimensionality in order for the lda to work
d_pca, e_pca = scipy.linalg.eigh(cov_tot, eigvals=(dim-200, dim-1))
x1_pca = x1.dot(e_pca)
x2_pca = x2.dot(e_pca)

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
mean_x1, cov_x1 = train_gauss(x1_lda)
mean_x2, cov_x2 = train_gauss(x2_lda)

total = 0
ok = 0
score = 0
"""
for filename, data in eval_data:
    data = crop_image(data, 2, 2, 2, 2)
    # standardise the data
    data = data.flatten()
    data -= train_mean
    data /= train_std

    data = (data.dot(e_pca)).dot(e_lda) # transform the test data
    ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
    ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))

    # indicates whether the data point was correctly classified
    result = sum(ll_target) - sum(ll_ntarget)
    if result >= 5.0:
        correct = True
        print(correct, result, filename)
    else: correct = False
          
    #print(correct, result, filename)
    """

print('======Target test data evaluation======')
for filename, data in test_target:
    total += 1
    data = crop_image(data, 2, 2, 2, 2)
    # standardise the data
    data = data.flatten()
    data -= train_mean
    data /= train_std
    
    data = (data.dot(e_pca)).dot(e_lda) # transform the test data
    ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
    ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))

    # indicates whether the data point was correctly classified
    result = sum(ll_target) - sum(ll_ntarget)
    score += result
    if result > THRESHOLD:
        ok += 1
        correct = True
    else: correct = False
          
    #print(correct, result)
    print(correct, result, filename)

print((ok/total) * 100, score)

print('')
print('======Non-target test data evaluation======')
total = 0
ok = 0
score = 0
for filename, data in test_ntarget:
    total += 1
    data = crop_image(data, 2, 2, 2, 2)
    # standardise the data
    data = data.flatten()
    data -= train_mean
    data /= train_std
    
    data = (data.dot(e_pca)).dot(e_lda)
    ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
    ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))
    
    # indicates whether the data point was correctly classified
    result = sum(ll_target) - sum(ll_ntarget)
    score += result
    if result <= THRESHOLD:
        ok += 1
        correct = True
    else: correct = False
          
    #print(correct, result)
    print(correct, result, filename)

print((ok/total) * 100, score)
