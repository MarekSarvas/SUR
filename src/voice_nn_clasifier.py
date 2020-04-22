import matplotlib.pyplot as plt
from ikrlib import *
import numpy as np
from numpy.random import randn


TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'

#train_t = list(wav16khz2mfcc(TRAIN_TARGET).values()) # target train data
#train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values()) # non-target train data

#print('TEST DATA')
test_t = wav16khz2mfcc(TEST_TARGET) # target test data
test_n = wav16khz2mfcc(TEST_NTARGET) # non-target test data

# convert to numpy arrays
#train_t = np.vstack(train_t)
#train_n = np.vstack(train_n)

test_n = np.vstack(list(test_n.values())[0])
test_t = np.vstack(list(test_t.values())[5])
#print(test[0].shape)

#t1 = np.ones(len(train_t))
#t2 = np.zeros(len(train_n))

#mu = np.mean(np.r_[train_t, train_n], axis=0)
#sig = np.std(np.r_[train_t, train_n], axis=0)

#train_t = (train_t - mu) / sig
#train_n = (train_n - mu) / sig


#x = np.r_[train_t, train_n]
#t = np.r_[t1, t2]

w1 = np.array([[-4.68193333, -4.65369809, -3.53167416,  7.3737802,  -0.95926537],
 [ 0.71863298 , 1.043952  ,-0.26285292 ,-1.3874706  ,-1.5113675 ],
 [ 0.45098247 ,-0.03087375,  2.44199613, -0.79778018, -2.2028055 ],
 [-2.4811921  , 0.92056346,  0.56977133, -0.91584511, -0.16884998],
 [ 0.44947158 ,-1.38394556,  2.68980507,  0.4698422 , -2.14529236],
 [-0.93288098 , 2.04199598,  0.14949756,  1.24671974, -0.21357838],
 [ 0.86269984 ,-0.88592889,  1.24240481, -1.45630097, -1.26854316],
 [ 1.246341   ,-0.09353945,  0.93673182,  1.0060212 , -1.16729874],
 [-1.26706584 ,-2.10414269, -1.23108569, -1.54087866, -1.20057591],
 [ 1.68914977 , 1.62760928,  0.36138657,  1.21422948, -0.23470183],
 [-0.78522768 ,-1.9724581 ,  0.05640913, -0.70451414, -0.25321386],
 [-0.18484387 , 1.53902975,  0.91835547,  0.34738764, -1.00810742],
 [-2.76203134 , 0.34663152, -1.51158909, -0.33794485, -0.86761132],
 [ 1.50569333 , 0.73710795, -0.91508168, -0.14655749, -0.86385422]] )

w2 = np.array( [[ 1.51380387],
 [-2.49437197],
 [-2.67843823],
 [-2.24803501],
 [-2.76693871],
 [-1.6563369 ]])

print(np.exp(np.mean(eval_nnet(test_n, w1, w2))))
print(np.exp(np.mean(eval_nnet(test_t, w1, w2))))