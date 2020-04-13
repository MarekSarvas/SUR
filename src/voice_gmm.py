##
# @file   voice_gmm.py 
# @brief  An attempt for voice recognition classifier using gaussian mixture model
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
THRESHOLD = 100
COMPONENTS = 20

# load target and non target voice data
train_t = list(wav16khz2mfcc(TRAIN_TARGET).values()) # target train data
train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values()) # non-target train data

print('TEST DATA')
test_t = wav16khz2mfcc(TEST_TARGET) # target test data
test_n = wav16khz2mfcc(TEST_NTARGET) # non-target test data

# convert to numpy arrays
train_t = np.vstack(train_t)
train_n = np.vstack(train_n)


mu1, cov1 = train_gauss(train_t)
mu2, cov2 = train_gauss(train_n)
p1 = p2 = 0.5

# Train and test with GMM models with full covariance matrices
# number of gmm components 
m1 = COMPONENTS

# Initialize mean vectors to randomly selected data points from corresponding class
mus1 = train_t[randint(1, len(train_t), m1)]

# Initialize all covariance matrices to the same covariance matrices computed using
# all the data from the given class
covs1 = [cov1] * m1

# uniform distribution as initial guess for the weights
ws1 = np.ones(m1) / m1

m2 = COMPONENTS
mus2 = train_n[randint(1, len(train_n), m2)]
covs2 = [cov2] * m2
ws2 = np.ones(m2) / m2

hard_decision = lambda x: logpdf_gauss(x, mu1, cov1) + np.log(p1) > logpdf_gauss(x, mu2, cov2) + np.log(p2)

# train gmm for target and non-target data in 100 iterations
print('Training gmm...')
for i in range(200):
    ws1, mus1, covs1, ttl1_new = train_gmm(train_t, ws1, mus1, covs1)
    ws2, mus2, covs2, ttl2_new = train_gmm(train_n, ws2, mus2, covs2)


"""
ttl1_old = None
ttl2_old = None
first_iter = True
while True:
    ws1, mus1, covs1, ttl1_new = train_gmm(train_t, ws1, mus1, covs1)
    ws2, mus2, covs2, ttl2_new = train_gmm(train_n, ws2, mus2, covs2)

    if not first_iter:
        if abs(abs(ttl1_new) - abs(ttl1_old)) < 0.0001  and abs(abs(ttl2_new) - abs(ttl2_old)) < 0.0001:
            break   
    
    ttl1_old = ttl1_new
    ttl2_old = ttl2_new
    first_iter = False
"""    

hit = 0
total = 0

print('=========================target test data===========================')
for filename in test_t:
    total += 1
    data = np.vstack(test_t[filename])

    ll_t = logpdf_gmm(data, ws1, mus1, covs1)
    ll_n = logpdf_gmm(data, ws2, mus2, covs2)

    """
    if logpdf_gmm(data, ws1, mus1, covs1) + np.log(p1) > logpdf_gmm(data, ws2, mus2, covs2) + np.log(p2):
        print('HIT')
    else:
        print('MISS')
    """
    # evaluate if the data point was correctly classified
    if int(sum(ll_t)) - int(sum(ll_n)) > THRESHOLD:
        status = 'HIT'
        hit += 1
    else:
        status = 'MISS'
    print(filename,': ', int(sum(ll_t)) - int(sum(ll_n)), ' : ', status)

hit_t = hit/total * 100
hit = 0
total = 0

print('=========================non-target test data===========================')
for filename in test_n:
    total += 1
    data = np.vstack(test_n[filename])

    ll_t = logpdf_gmm(data, ws1, mus1, covs1)
    ll_n = logpdf_gmm(data, ws2, mus2, covs2)
    
    """
    if logpdf_gmm(data, ws1, mus1, covs1) + np.log(p1) > logpdf_gmm(data, ws2, mus2, covs2) + np.log(p2):
        print('HIT')
    else:
        print('MISS')
    """
    # evaluate if the data point was correctly classified
    if int(sum(ll_t)) - int(sum(ll_n)) > THRESHOLD:
        status = 'MISS'
    else:
        status = 'HIT'
        hit += 1

    print(filename,': ', int(sum(ll_t)) - int(sum(ll_n)), ' : ', status)

hit_n = hit/total * 100

print("Hit target data%: ", hit_t)
print("Hit non-targetdata %: ", hit_n)


write_new = input('Do you want to save trained gmm data ?(y/n): ')
if write_new.lower() == 'y':
    with open("gmm_values.txt", "w") as f:
        f.write('Target data:\n')
        f.write('ws: {0}\n'.format(ws1))
        f.write('mus: {0}\n'.format(mus1))
        f.write('covs: {0}\n'.format(covs1))
        f.write('Non-target data:\n')
        f.write('ws: {0}\n'.format(ws2))
        f.write('mus: {0}\n'.format(mus2))
        f.write('covs: {0}\n'.format(covs2))

