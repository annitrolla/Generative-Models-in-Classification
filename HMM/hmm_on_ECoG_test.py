# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""
from HMM.hmm_classifier import HMMClassifier
from DataNexus.datahandler import DataHandler
import numpy as np

train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(20, 100, train_data, train_labels)
print hmmcl.test(model_pos, model_neg, test_data, test_labels)

