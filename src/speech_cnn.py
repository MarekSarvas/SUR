#!/usr/bin/python3.7

##
# @file   speech_cnn.py 
# @brief  This classifier implements a cnn and uses it to recognize a target speaker
#         in the given data.

import tensorflow as tf
import numpy as np
import sys

from ikrlib import wav16khz2mfcc
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.utils import to_categorical

THRESHOLD = -30.0
PRINT_STATS = True
VALIDATE = False # determines, whether the model should train with validation data

# paths to data directories NOTE: edit these to suit your needs...
#==============================================================
# training data
TRAIN_TARGET = '../data/train_ultimate_target/'
TRAIN_NTARGET = '../data/train_ultimate_ntarget/'

# data meant for classification
EVAL_DATA = '../../SUR_projekt2019-2020_eval/eval/'
#==============================================================

TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'

# Some parameters for us to play with....
MEAN_SEGMENT_LEN = 10
INITIAL_CUTOFF = 100
MEAN_MULTIPLIER = 1
DEFAULT_MEAN = 40.0

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
        #print(np.mean(seg[:][:,0]), mean_energy, mean_energy*MEAN_MULTIPLIER)
        
        if (np.mean(seg[:][:,0]) > mean_energy * MEAN_MULTIPLIER):
            new.append(seg)
            
    return np.vstack(new)

SEGMENT_LEN = 50
STEP = 5

# this function creates "pictures" from our features by grouping them up
def create_frame_batches(data):
    # we need the speech sample to be at least as long as our batch length
    while data.shape[0] < SEGMENT_LEN:
        data += data
    
    grouped = []
    for i in range(0, data.shape[0] - SEGMENT_LEN, STEP):
        group = []
        for j in range(SEGMENT_LEN):
            group.append(data[i+j])
        grouped.append(np.vstack(group).flatten().reshape(SEGMENT_LEN, 13, 1))
    return np.array(grouped)


def train_classifier(validation=False):
    # First, load the data
    train_t = list(wav16khz2mfcc(TRAIN_TARGET).values()) # target train data
    train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values()) # non-target train data

    if validation:
        test_t = wav16khz2mfcc(TEST_TARGET) # target test data
        test_n = wav16khz2mfcc(TEST_NTARGET) # non-target test data

    target = []
    for rec in train_t:
        target.append(remove_silence(rec))

    ntarget = []
    for rec in train_n:
        ntarget.append(remove_silence(rec))

    X_train_t = np.vstack(target)
    X_train_n = np.vstack(ntarget)

    if validation:
        test_target = []
        for rec in list(test_t.values()):
            test_target.append(remove_silence(rec))

        test_ntarget = []
        for rec in list(test_n.values()):
            test_ntarget.append(remove_silence(rec))

        X_test_t = np.vstack(test_target)
        X_test_n = np.vstack(test_ntarget)

    # Create 13x13 batches from the data
    X_train_t = create_frame_batches(X_train_t)
    X_train_n = create_frame_batches(X_train_n)

    if validation:
        X_test_t = create_frame_batches(X_test_t)
        X_test_n = create_frame_batches(X_test_n)

    # Get all the data to one place
    X_train = np.vstack((X_train_t, X_train_n))
    y_train = np.hstack((np.zeros(X_train_t.shape[0]), np.ones(X_train_n.shape[0])))
    y_train_hot = to_categorical(y_train)

    if validation:
        X_test = np.vstack((X_test_t, X_test_n))
        y_test = np.hstack((np.zeros(X_test_t.shape[0]), np.ones(X_test_n.shape[0])))
        y_test_hot = to_categorical(y_test)

    num_classes = 2

    # Now build the model
    model = Sequential()
    model.add(Conv2D(16, (3, 3),
        input_shape=(SEGMENT_LEN, 13, 1),
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=['accuracy'])

    if validation:
        model.fit(X_train, y_train_hot, batch_size=5, epochs=5, validation_data=(X_test, y_test_hot))
    else:
        model.fit(X_train, y_train_hot, batch_size=5, epochs=5)

    model.summary()
    model.save('cnn_speech_trained_model.h5')

# returns a tuple of (class, score), where class is '1' for the
# target and '0' otherwise.
def classify_record(model, record):
    rec = remove_silence(record)

    X = create_frame_batches(rec)
    predict = model.predict(X)
    target = 0
    ntarget = 0
    for val in predict:
        if val[0] > val[1]:
            target += 1
        else:
            ntarget += 1

    #score = sum(predict[:][:,0])*target - sum(predict[:][:,1])*ntarget
    score = target - ntarget
    cls = 1 if score >= THRESHOLD else 0
    return cls, score

def classify():
    # load model
    try:
        model = keras.models.load_model('cnn_speech_trained_model.h5')
    except:
        print('Error: No model file found, please train the classifier first.')
        sys.exit()

    # load the evaluation data
    eval_data = list(wav16khz2mfcc(EVAL_DATA).items())

    # ...aaand classify the given data
    count = 0
    results = []
    for filename, data in eval_data:

        # compute the score of the picture
        cls, score = classify_record(model, data)
        results.append(filename.split('/')[-1].split('.')[0]
                + ' ' + str(score)
                + ' ' + str(cls))
        if cls: count += 1 # just remember we got a target...

    results.sort()
    output = open('audio_convolutionalNN.txt', 'w')
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
    if train: train_classifier(VALIDATE)
    if evaluate: classify()
