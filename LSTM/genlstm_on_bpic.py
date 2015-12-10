"""

Classifier based on generative LSTM applied to bpic1

"""

import numpy as np
from keras.optimizers import RMSprop
from LSTM.lstm_classifier import LSTMClassifier

# parameters
nfolds = 5

lstmsize = 500
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 50
lstmbatch = 512

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')

# Merge train and test
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # train the model and report performance
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatch, validation_split=0.3)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
    scores.append(lstmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx]))

print 'Generative LSTM classifier on dynamic features: %.4f (+- %.4f) %s' % (np.mean(scores), np.std(scores), scores)

