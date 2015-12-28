"""

Train the whole set of test on the "lstm wins" synthetic dataset
 

"""

import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier 

# general parameters
nfolds = 5
lstmsize = 1000
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 50
lstmbatchsize = 64

#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')


#
# Merge train and test
#
print "Combining data from train and test for CV predictions..."
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]


#
# Prepare combined datasets for the future experiments
#

# dataset to check how generative models perform if provided with static features along with dynamic
static_as_dynamic = np.zeros((static_all.shape[0], static_all.shape[1], dynamic_all.shape[2]))
for i in range(static_all.shape[0]):
    static_as_dynamic[i, :, :] = np.tile(static_all[i, :], (dynamic_all.shape[2], 1)).T
dynamic_and_static_as_dynamic = np.concatenate((dynamic_all, static_as_dynamic + np.random.uniform(-0.0001, 0.0001, static_as_dynamic.shape)), axis=1)


#
# k-fold CV for performance estimation
#
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # LSTM on dynamic and static (turned into fake sequences) (10)
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_and_static_as_dynamic[train_idx], labels_all[train_idx])
    scores[10].append(lstmcl.test(model_pos, model_neg, dynamic_and_static_as_dynamic[val_idx], labels_all[val_idx]))

print "===> (10) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[10]), np.std(scores[10]), scores[10])


