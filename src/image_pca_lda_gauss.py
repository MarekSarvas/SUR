#!/usr/bin/python3.7

##
# @file   image_pca_lda_gauss.py 
# @brief  A shaky face recognition classifier. Utilizes pca, lda, and a gaussian
#         probability distribution.

import numpy as np
import sys
import pickle
import scipy
import imageio

from ikrlib import png2fea, train_gauss, logpdf_gauss
from scipy.ndimage import rotate

THRESHOLD = 2.0 # the hard decision threshold
PRINT_STATS = True

# paths to data directories NOTE: edit these to suit your needs...
#==============================================================
# training data
TRAIN_TARGET = '../data/train_ultimate_target/'
TRAIN_NTARGET = '../data/train_ultimate_ntarget/'

# data meant for classification
EVAL_DATA = '../../SUR_projekt2019-2020_eval/eval/'
#==============================================================

# The folowing four functions are used in the data augmentation process
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


def train_classifier():
    # first, load target and non-target training data
    train_target = png2fea(TRAIN_TARGET) # target training data
    train_ntarget = png2fea(TRAIN_NTARGET) # non-target training data

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

    # PCA - reduce the dimensionality
    d_pca, e_pca = scipy.linalg.eigh(cov_tot, eigvals=(dim-150, dim-1))
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

    # compute the gaussian distributions for our classes and evalueate the test data
    mean_x1, cov_x1 = train_gauss(x1_lda)
    mean_x2, cov_x2 = train_gauss(x2_lda)

    # now save the model parameters
    model_parameters = [e_pca, e_lda, (mean_x1, cov_x1), (mean_x2, cov_x2),
            train_mean, train_std]
    with open('img_pca_lda.model', 'wb') as f:
        pickle.dump(model_parameters, f)
        f.close()


# This function classifies the data, path to which is stored in the EVAL_DATA
# constant. The name of each file and it's score as well as the hard
# decision is printed out once the file is classified
def classify():
    try:
        f = open('img_pca_lda.model', 'rb')
    except IOError:
        print('Error: File with model parameters not found, please train the ',
              + 'classifier first.')
        sys.exit()

    # load the model parameters
    model_parameters = pickle.load(f)
    e_pca = model_parameters[0]
    e_lda = model_parameters[1]
    mean_x1 = model_parameters[2][0]
    cov_x1 = model_parameters[2][1]
    mean_x2 = model_parameters[3][0]
    cov_x2 = model_parameters[3][1]
    train_mean = model_parameters[4]
    train_std = model_parameters[5]

    # load the evaluation data
    eval_data = list(png2fea(EVAL_DATA).items()) # evaluation data

    # ...aaand classify the given data
    count = 0
    results = []
    for filename, data in eval_data:
        data = crop_image(data, 2, 2, 2, 2)
        # standardise the data
        data = data.flatten()
        data -= train_mean
        data /= train_std

        data = (data.dot(e_pca)).dot(e_lda) # transform the test data

        # compute the log-likelihood for the classified picture
        ll_target = logpdf_gauss(data, mean_x1, np.atleast_2d(cov_x1))
        ll_ntarget = logpdf_gauss(data, mean_x2, np.atleast_2d(cov_x2))

        # compute the score of the picture
        score = sum(ll_target) - sum(ll_ntarget)

        # make the hard decision
        if score >= THRESHOLD: decision = 1; count += 1 # target
        else: decision = 0 # non-target

        results.append(filename.split('/')[-1].split('.')[0]
                + ' ' + str(np.around(score, decimals=8))
                + ' ' + str(decision))

    results.sort()
    output = open('image_pca_lda_gaussian.txt', 'w')
    for result in results: print(result, file=output)
    if PRINT_STATS: print(f'Targets found: {count}')
    output.close()

if __name__ == "__main__":
    # First, parse the program args..

    train = False
    evaluate = False

    if '--train' in sys.argv: train = True
    if '--eval' in sys.argv: evaluate = True
    if train == False and evaluate == False:
        print('Please specify whether you want to train the classifier by using the '
              + '[--train] argument, or whether you want to classify some data, by using the '
              + '[--eval] parameter')

    # launch the specified functions
    if train: train_classifier()
    if evaluate: classify()
