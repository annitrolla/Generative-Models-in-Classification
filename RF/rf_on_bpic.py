# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# parameters
nfolds = 5
nestimators = 500

static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_test = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_test = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_test = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# dataset to confirm that RF on dynamic is not better than generative models on dynamic data
dynamic_as_static = dynamic_train.reshape((dynamic_train.shape[0], dynamic_train.shape[1] * dynamic_train.shape[2]))

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores_acc = []
for fid, val_idx in enumerate(val_idx_list):
    print "Running fold %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    # RF on dynamic features (6)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(dynamic_as_static[train_idx], labels_train[train_idx]) 
    scores_acc.append(rf.score(dynamic_as_static[val_idx], labels_train[val_idx]))

print "===> (6) RF on dynamic (spatialized) features: %.4f (+/- %.4f) %s" % (np.mean(scores_acc), np.std(scores_acc), scores_acc)

