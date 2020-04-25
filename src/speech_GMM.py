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
import pickle
from scipy.ndimage import gaussian_filter

# paths to data directories
TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/' 
TEST_NTARGET = '../data/non_target_dev/'
EVAL_DATA = '../eval/' # default var used in evaluation

# Some parameters for us to play with....
MEAN_SEGMENT_LEN = 20
INITIAL_CUTOFF = 190
MEAN_MULTIPLIER = 1
DEFAULT_MEAN = 40.0
COMPONENTS = 20


# this function cuts up the array to chunks and lets us process these
def divide_chunks(l, n):
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]
        
def remove_silence(record):
    # First cut off the first 190 frames of the recording
    record = record[INITIAL_CUTOFF:]
    # calculate the mean energy in order to remove silence
    mean_energy = np.mean(record[:][:,0])
    
    if mean_energy > DEFAULT_MEAN: mean_energy = DEFAULT_MEAN
    
    # now split the arrays into segments of length MEAN_SEGMENT_LEN
    # and compare the mean of these chunks to the overall mean
    new = []
    for seg in divide_chunks(record, MEAN_SEGMENT_LEN):
        
        if np.mean(seg[:][:,0]) > mean_energy * MEAN_MULTIPLIER:
            new.append(seg)
            
    return np.vstack(new)

# load trained values for gmm clasification
def load_gmm_params():
    model_parameters = []
    with open('speech_gmm.model', 'rb') as f:
        model_parameters = pickle.load(f)

    return tuple(model_parameters)

def modify_filename(filename):
    filename = filename.split('/')[-1]
    return filename.split('.wav')[0]

# evaluate .wav files in data_path folder using trained gmm
def evaluate_speech_gmm(data_path=EVAL_DATA):
    THRESHOLD = 100
    (ws1, mus1, covs1, ws2, mus2, covs2) = load_gmm_params()
    test_data = wav16khz2mfcc(data_path)
    hit = 0
    total = 0
    result = []
    for filename in test_data:
        # remove silence from test data
        data = remove_silence(test_data[filename])
        total += 1

        # evaluate log likelihood
        ll_t = logpdf_gmm(data, ws1, mus1, covs1)
        ll_n = logpdf_gmm(data, ws2, mus2, covs2)

        # evaluate if the data point was correctly classified
        if sum(ll_t) - sum(ll_n) > THRESHOLD:
            status = 1
            hit += 1
        else:
            status = 0
        result.append("{}  {:.10f}  {}".format(modify_filename(filename), sum(ll_t) - sum(ll_n), status))
    
    with open('speech_GMM.txt', 'w') as f:
        for item in sorted(result):
            f.write(item+"\n")
            
  
def my_train_gmm():

    # load target and non target voice data
    train_t = list(wav16khz2mfcc(TRAIN_TARGET).values()) # target train data
    train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values()) # non-target train data
    test_t = list(wav16khz2mfcc(TEST_TARGET).values())
    test_n = list(wav16khz2mfcc(TEST_NTARGET).values())
    
    # remove silence from train data
    new_t = []
    for rec in train_t:
        new_t.append(remove_silence(rec))
    for rec in test_t:
        new_t.append(remove_silence(rec))

    new_n = []
    for rec in train_n:
        new_n.append(remove_silence(rec))

    for rec in test_n:
        new_n.append(remove_silence(rec))

    train_t = new_t
    train_n = new_n

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

    # train gmm for target and non-target data in 100 iterations
    print('Training gmm...')
    for i in range(200):
        ws1, mus1, covs1, ttl1_new = train_gmm(train_t, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2_new = train_gmm(train_n, ws2, mus2, covs2)

    a = input("Do you want to save gmm values?(Y/n)")
    if str(a).upper() == 'Y':
        model_parameters = [ws1, mus1, covs1, ws2, mus2, covs2]
        with open('speech_GMM.model', 'wb') as f:
            pickle.dump(model_parameters, f)
            f.close()

        
if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("Invalid number of arguments, try --help")
    sys.exit(1)
if sys.argv[1] == '--help':
    print("--eval argument for evaluating data on trained gmm for speech clasification\n--train for training gmm for speech clasification")
elif sys.argv[1] == '--eval':
    evaluate_speech_gmm()
elif sys.argv[1] == '--train':
    my_train_gmm()
elif (sys.argv[1] == '--eval' and sys.argv[2] == '--train') or (sys.argv[2] == '--eval' and sys.argv[1] == '--train'):
    my_train_gmm()
    evaluate_speech_gmm()
else:
    print("Invalid program argument, try --help")
    sys.exit(1)