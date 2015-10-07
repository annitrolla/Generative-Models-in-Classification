# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""

from HMM.hmm_classifier import HMMClassifier
import numpy as np
import cPickle

# load the dataset
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy")

# initialize HMM based classifier
hmmcl = HMMClassifier()

# train a model pair for each feature
models = hmmcl.train_per_feature(30, 5, train_data, train_labels)

# store the models
with open('../../Results/models/hmm_per_feature.pkl', 'w') as f:
    cPickle.dump(models, f)

# show accuracy on the test set
print hmmcl.test_per_feature(models, test_data, test_labels)

