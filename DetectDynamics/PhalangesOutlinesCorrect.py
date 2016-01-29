"""

Train the whole set of test on the "lstm wins" synthetic dataset

"""

import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
from LSTM.lstm_classifier import LSTMClassifier 

# general parameters
nfolds = 5
nestimators = 500
nhmmstates = 6
nhmmiter = 50
hmmcovtype = "full"  # options: full, diag, spherical
lstmsize = 200
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 20
lstmbatchsize = 256


#
# Load the dataset
#
print 'Loading the dataset..'
dynamic_all = np.load('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/preprocessed/train_dynamic.npy')
labels_all = np.load('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/preprocessed/train_labels.npy')
nsamples = dynamic_all.shape[0]

# dataset to confirm that RF on dynamic is not better than generative models on dynamic data
dynamic_as_static = dynamic_all.reshape((dynamic_all.shape[0], dynamic_all.shape[1] * dynamic_all.shape[2]))


#
# k-fold CV for performance estimation
#
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # RF on dynamic features (6)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(dynamic_as_static[train_idx], labels_all[train_idx])
    scores[6].append(rf.score(dynamic_as_static[val_idx], labels_all[val_idx]))

    # HMM on dynamic features (7)
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, dynamic_all[train_idx], labels_all[train_idx])
    acc, auc = hmmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx])
    scores[7].append(acc)

    # LSTM on dynamic features (8)
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
    scores[8].append(lstmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx]))


print "===> (6) RF on dynamic (spatialized) features: %.4f (+/- %.4f) %s" % (np.mean(scores[6]), np.std(scores[6]), scores[6])
print "===> (7) HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[7]), np.std(scores[7]), scores[7])
print "===> (8) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[8]), np.std(scores[8]), scores[8])

