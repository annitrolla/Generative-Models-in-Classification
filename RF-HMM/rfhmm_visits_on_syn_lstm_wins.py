"""

RF on static features enriched by HMM. From HMM we extract number of times each HMM state
was visited under each of the models.

"""

import numpy as np
from DataNexus.datahandler import DataHandler 
from DataNexus.fourier import Fourier
from HMM.hmm_classifier import HMMClassifier 
from sklearn.ensemble import RandomForestClassifier
import cPickle


#
# Parameters
#
nstates = 3
niter = 2
covtype = 'full'
nfolds = 5
nestimators = 300


#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')


#
# Merge train and test
#
print "Combining data from train and test for CV predictions..."
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]


#
# Cross-validation to collect enrichment features
#
visits_all_hmm = np.empty((len(labels_all), nstates * 2))

predict_idx_list = np.array_split(range(nsamples), nfolds)
for fid, predict_idx in enumerate(predict_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(predict_idx))
    
    # extract visit counts from HMM on dynamic
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nstates, niter, covtype, dynamic_all[train_idx], labels_all[train_idx])
    visits_pos, visits_neg = hmmcl.state_visits(model_pos, model_neg, dynamic_all[predict_idx])
    visits_all_hmm[predict_idx] = np.hstack((visits_pos, visits_neg))

# prepare the dataset
enriched_by_hmm = np.concatenate((static_all, visits_all_hmm), axis=1)

# split the data into training and test
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

# train hybrid on features enriched by HMM (3)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
print "===> (3) Hybrid (RF) on features (state visit counts) enriched by HMM: %.4f" % rf.score(enriched_by_hmm[test_idx], labels_all[test_idx])

