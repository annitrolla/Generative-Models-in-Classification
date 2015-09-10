# -*- coding: utf-8 -*-
"""
handles functions required for hmm classification
"""

import numpy as np
from hmmlearn import hmm


class HMMClassifier:
    
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
    
    def find_best_parameter(self, ratio, hdn_nstates_list, niter, nrepetitions, data, labels):
        """
        parameter search over number of hidden states
        @param hdn_nstates_list: list of number of hidden states to try, e.g. range(2,10)
        @param ratio: ratio of the dataset split for train, e.g. 0.7
        @param niter: number of iterations for hmm model to perform, e.g. 10
        @param nrepetitions: number of repeated runs for the same hidden state, but for the different split, e.g. 5
        """
        accuracy_results = {}
        for nstates in hdn_nstates_list:
            print 'state' + str(nstates)
            accuracy_results[nstates] = []
            for run in range(nrepetitions):
                print 'repetition' + ' ' + str(run) 
                
                # make new random split  
                train_data, train_labels, val_data, val_labels = self.dh.split_train(ratio, data, labels)
                
                # train a model on this split
                model_pos, model_neg = self.train(nstates, niter, train_data, train_labels)
                
                # test the model and store the results
                acc = self.test(model_pos, model_neg, val_data, val_labels)
                print nstates, acc
                accuracy_results[nstates].append(acc)

        with open("../../Results/crossvalidated_accuracy.txt","w") as f:
            for nstates in hdn_nstates_list:
                print nstates, np.mean(accuracy_results[nstates]), np.std(accuracy_results[nstates])
                f.write("%d, %s\n" % (nstates, ", ".join([str(x) for x in accuracy_results[nstates]])))
    
    def train(self, nstates, niter, data, labels):
        train_pos = self.tensor_to_list(data[labels==1, :, :])
        train_neg = self.tensor_to_list(data[labels==0, :, :])
        print "Start training the positive model..."
        model_pos = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
        model_pos.fit(train_pos)
        print "Start training the negative model..."
        model_neg = hmm.GaussianHMM(nstates, covariance_type="full", n_iter=niter)
        model_neg.fit(train_neg)
        return model_pos, model_neg

    def test(self, model_pos, model_neg, data, labels):
        test = self.tensor_to_list(data)
        return self.accuracy(test, labels, model_pos, model_neg)

    def pos_neg_ratios(self, model_pos, model_neg, data):
        data = self.tensor_to_list(data)
        # find log-likelihood of positive model for a new sequence and the same for a negative one, substract
        ratios = np.empty(len(data))
        for i in range(len(data)):
            ratios[i] = model_pos.score(data[i]) - model_neg.score(data[i])
        return ratios

