# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""

from HMM.hmm_classifier import HMMClassifier
import numpy as np

# load the dataset
train_data = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy")

# TEMPORARY FOR TESTING PURPOSES
#train_data = train_data[:, 0:2, :]
#test_data = test_data[:, 0:2, :]

# initialize HMM based classifier
hmmcl = HMMClassifier()
models_pos, models_neg = hmmcl.train_per_feature(3, 10, train_data, train_labels)
print hmmcl.test_per_feature(models_pos, models_neg, test_data, test_labels)

