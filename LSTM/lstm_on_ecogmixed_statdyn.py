"""

We feed static features along with dynamical ones into LSTM.
Statical feature is replicated [seqlen] times to be transformed into a "sequence", where value
does not change over time. Kind of "fake" sequence.

"""

import numpy as np
from LSTM.lstm_classifier import LSTMClassifier 

# general parameters
lstm_nepochs = 20

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')

# transform static features into "fake" sequences
dynamized_static_train = np.zeros((static_train.shape[0], static_train.shape[1], dynamic_train.shape[2]))
for i in range(static_train.shape[0]):
    dynamized_static_train[i, :, :] = np.tile(static_train[i, :], (dynamic_train.shape[2], 1)).T
dynamized_static_val = np.zeros((static_val.shape[0], static_val.shape[1], dynamic_val.shape[2]))
for i in range(static_val.shape[0]):
    dynamized_static_val[i, :, :] = np.tile(static_val[i, :], (dynamic_val.shape[2], 1)).T

# meld dynamized static and dynamic features together
all_train = np.concatenate((dynamized_static_train, dynamic_train), axis=1)
all_val = np.concatenate((dynamized_static_val, dynamic_val), axis=1)

# dynamic data with LSTM
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(all_train, labels_train)
print "LSTM with dynamized static and dynamic features on validation set: %.4f" % lstmcl.test(model_pos, model_neg, all_val, labels_val)




