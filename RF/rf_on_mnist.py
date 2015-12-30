# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# parameters
nfolds = 5
nestimators = 500

static_train = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_dynamic.npy')
static_test = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_static.npy')
dynamic_test = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_labels.npy')
labels_test = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores_acc = []
for fid, val_idx in enumerate(val_idx_list):
    print "Running fold %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    # RF on dynamic features (6)
    rf = RandomForestClassifier(n_estimators=nestimators, n_jobs=-1)
    rf.fit(static_train[train_idx], labels_train[train_idx]) 
    scores_acc.append(rf.score(static_train[val_idx], labels_train[val_idx]))

print "===> (5) RF on static features: %.4f (+/- %.4f) %s" % (np.mean(scores_acc), np.std(scores_acc), scores_acc)

