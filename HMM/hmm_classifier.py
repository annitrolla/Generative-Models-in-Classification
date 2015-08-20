# -*- coding: utf-8 -*-
"""
handles functions required for hmm classification
"""

import numpy as np
from hmmlearn import hmm
from Preprocessing.ecog import ECoG

class HMMClassifier:
    
    # load the data
    ecog = None
    
    def __init__(self):
        self.ecog = ECoG()
        self.ecog.load_train_data()
        
    
    # tensor to list (for HMM input)
    def tensor_to_list(self, tensor):
        """
        from [SUBJECTS, TIME, FEATURES] to   
        list[SUBJECT] = matrix[TIME, FEATURES]
        """
  
        lst = []
        for i in range(0, tensor.shape[0]):
            lst.append(tensor[i, :, :].T)
        return lst
        
    # building Maximum likelihood classifier
    def ispositive(self, instance, model_pos, model_neg):
        return model_pos.score(instance) >= model_neg.score(instance)

    def accuracy(self, data, labels, model_pos, model_neg, show_prediction=False):
        pred = []
        for i in range(len(data)):
            pred.append(int(self.ispositive(data[i], model_pos, model_neg)))
        acc = float(sum(pred == labels))/float(len(pred))
        if show_prediction == True:
            return acc, pred
        else:
            return acc
    
    def find_best_parameter(self, ratio, hdn_nstates_list, niter, nrepeatitions):
        """
        parameter search over number of hidden states
        @param hdn_nstates_list: list of number of hidden states to try, e.g. range(2,10)
        @param ratio: ratio of the dataset split for train, e.g. 0.7
        @param niter: number of iterations for hmm model to perform, e.g. 10
        @param nrepeatitions: number of repeated run for the same hidden state, but for the different split, e.g. range(5)
        """
        accuracy_results = {}
        for nstates in hdn_nstates_list:
            accuracy_results[nstates] = []
            for run in nrepeatitions:
                # make new random split  
                train_data, train_labels, val_data, val_labels = self.ecog.split_train_val(ratio)
                train_pos = train_data[train_labels==1, :, :]
                train_neg = train_data[train_labels==0, :, :]
                train_pos = self.tensor_to_list(train_pos)
                train_neg = self.tensor_to_list(train_neg)
                val_data = self.tensor_to_list(val_data)
                model_pos = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
                model_pos.fit(train_pos)
                model_neg = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
                model_neg.fit(train_neg)
                # validation
                acc = self.accuracy(val_data, val_labels, model_pos, model_neg)
                print nstates, acc
                accuracy_results[nstates].append(acc)

        with open("../../Results/crossvalidated_accuracy.txt","w") as f:
            for nstates in hdn_nstates_list:
                print nstates, np.mean(accuracy_results[nstates]), np.std(accuracy_results[nstates])
                f.write("%d, %s\n" % (nstates, ", ".join([str(x) for x in accuracy_results[nstates]])))
    
    def test_model(self, nstates, niter):
        self.ecog.load_test_data()
        
        trainval_pos = self.tensor_to_list(self.ecog.trainval_data[self.ecog.trainval_labels==1, :, :])
        trainval_neg = self.tensor_to_list(self.ecog.trainval_data[self.ecog.trainval_labels==0, :, :])
        test = self.tensor_to_list(self.ecog.test_data)
        model_pos = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
        model_pos.fit(trainval_pos)
        model_neg = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
        model_neg.fit(trainval_neg)
        acc = self.accuracy(test, self.ecog.test_labels, model_pos, model_neg)
        return acc, nstates
