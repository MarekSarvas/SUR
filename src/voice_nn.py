import matplotlib.pyplot as plt
from ikrlib import *
import numpy as np
from numpy.random import randn


TRAIN_TARGET = '../data/target_train/'
TRAIN_NTARGET = '../data/non_target_train/'
TEST_TARGET = '../data/target_dev/'
TEST_NTARGET = '../data/non_target_dev/'

train_t = list(wav16khz2mfcc(TRAIN_TARGET).values()) # target train data
train_n = list(wav16khz2mfcc(TRAIN_NTARGET).values()) # non-target train data

#print('TEST DATA')
test_t = wav16khz2mfcc(TEST_TARGET) # target test data
test_n = wav16khz2mfcc(TEST_NTARGET) # non-target test data

# convert to numpy arrays
train_t = np.vstack(train_t)
train_n = np.vstack(train_n)

test = np.vstack(list(test_t.values())[0])
#print(test[0].shape)

t1 = np.ones(len(train_t))
t2 = np.zeros(len(train_n))

mu = np.mean(np.r_[train_t, train_n], axis=0)
sig = np.std(np.r_[train_t, train_n], axis=0)

train_t = (train_t - mu) / sig
train_n = (train_n - mu) / sig


x = np.r_[train_t, train_n]
t = np.r_[t1, t2]



dim_in = 13
dim_hidden = 6
dim_out = 1

w1 = randn(dim_in + 1, dim_hidden) * .1
w2 = randn(dim_hidden + 1, dim_out) * .1
print("w2: ",w2.shape)
epsilon = .05

for i in range(10):
    """
    plot2dfun(lambda x: eval_nnet(x, w1, w2), ax, 100)
    plt.plot(x1[:,0], x1[:,1], 'rx', x2[:,0], x2[:,1], 'bx')
    plt.show()
    """
    w1, w2, ed = train_nnet(x, t, w1, w2, epsilon)
    #print('Total log-likelihood: %f' % -ed)

print(w1, w2)