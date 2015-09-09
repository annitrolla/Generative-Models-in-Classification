# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""
from HMM.hmm_classifier import HMMClassifier
from DataNexus.datahandler import DataHandler


dh = DataHandler("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed")
dh.load_train_data()
dh.load_test_data()

hmmcl = HMMClassifier(dh)
#print hmmcl.test_model(2, 10)
hmmcl.find_best_parameter(0.7, range(10,21), 10, 5)
