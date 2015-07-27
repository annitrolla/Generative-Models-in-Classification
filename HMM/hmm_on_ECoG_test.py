# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:58:39 2015

@author: annaleontjeva
"""

import numpy as np
from hmmlearn import hmm
from Preprocessing.ecog import ECoG

# import matplotlib.pyplot as plt

# load the data
ecog = ECoG() 
ecog.load_train_data() 
ecog.load_test_data()

def class_division(data, labels):
    pos = data[labels==1, :, :]
    neg = data[labels==0, :, :]
    return pos, neg
    
# tensor to list (for HMM input)
def tensor_to_list(tensor):
    """
    from [SUBJECTS, TIME, FEATURES] to   
    list[SUBJECT] = matrix[TIME, FEATURES]
    """    
    lst = []
    for i in range(0, tensor.shape[0]):
        lst.append(tensor[i, :, :].T)
    return lst
 
# Building Maximum likelihood classifier
def ispositive(model_pos, model_neg, instance):
    return model_pos.score(instance) >= model_neg.score(instance)

def accuracy(data, labels, model_pos, model_neg, show_prediction=False):
    pred = []
    for i in range(len(data)):
        pred.append(int(ispositive(model_pos, model_neg, data[i])))
    acc = float(sum(pred == labels))/float(len(pred))
    if show_prediction == True:
        return acc, pred
    else:
        return acc
        
# parameter search over number of hidden states
hidden_state_range = range(2,20)
accuracy_results = {}
for nstates in hidden_state_range:
    accuracy_results[nstates] = []
    for run in range(10):
        # make new random split    
        train_pos, train_neg, val_data, val_labels = random_split(data=trainval_data, labels=trainval_labels, ratio=0.7)
        model_pos = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=50)        
        model_pos.fit(train_pos)
        model_neg = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=50)
        model_neg.fit(train_neg)   
        # validation
        acc = accuracy(val_data, val_labels, model_pos, model_neg)
        print nstates, acc
        accuracy_results[nstates].append(acc)

with open("../../Results/crossvalidated_accuracy.txt","w") as f:
    for nstates in hidden_state_range:
        print nstates, np.mean(accuracy_results[nstates]), np.std(accuracy_results[nstates])
        f.write("%d, %s\n" % (nstates, ", ".join([str(x) for x in accuracy_results[nstates]])))

# Now use real test data
# model_pos = hmm.GaussianHMM(best_state_number, covariance_type="full", n_iter=1000)
# trainval_pos, trainval_neg, _, _ = random_split(trainval_data, trainval_labels, ratio=1.0)
# model_pos.fit(trainval_pos)
# model_neg = hmm.GaussianHMM(best_state_number, covariance_type="full", n_iter=1000)
# model_neg.fit(trainval_neg)
