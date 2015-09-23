# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""

from HMM.hmm_classifier import HMMClassifier
import numpy as np

train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy")

hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(30, 5, train_data, train_labels)
print hmmcl.test(model_pos, model_neg, test_data, test_labels)

