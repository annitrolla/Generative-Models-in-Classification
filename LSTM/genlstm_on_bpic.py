"""

Classifier based on generative LSTM applied to syn_lstm_wins

"""

import numpy as np
from keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder
from LSTM.lstm_classifier import LSTMClassifier

# parameters
nfolds = 5

lstmsize = 2000
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 20
lstmbatch = 64

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # train the model and report performance
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatch)
    model_pos, model_neg = lstmcl.train(dynamic_train[train_idx], labels_train[train_idx])
    scores.append(lstmcl.test(model_pos, model_neg, dynamic_train[val_idx], labels_train[val_idx]))

print 'Generative LSTM classifier on dynamic features: %.4f (+- %.4f) %s' % (np.mean(scores), np.std(scores), scores)

