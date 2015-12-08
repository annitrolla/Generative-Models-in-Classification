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
nestimators = 300
nhmmstates = 3
nhmmiter = 10
hmmcovtype = "full"  # options: full, diag, spherical

#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')


#
# Merge train and test
#
print "Combining data from train and test for CV predictions..."
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]


#
# k-fold CV
#


# prepare where to store the ratios
ratios_all_hmm = np.empty(len(labels_all))

# split indices into folds
enrich_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
for fid, enrich_idx in enumerate(enrich_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(enrich_idx))

    # extract predictions using HMM on dynamic
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                       dynamic_all[train_idx], labels_all[train_idx])
    ratios_all_hmm[enrich_idx] = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[enrich_idx])

#
# Prepare combined datasets for the future experiments
#

# datasets for hybrid learning
enriched_by_hmm = np.concatenate((static_all, np.matrix(ratios_all_hmm).T), axis=1)


#
# (2.) k-fold cross validation to obtain accuracy
#
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    # Hybrid on features enriched by HMM (3)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
    scores.append(rf.score(enriched_by_hmm[val_idx], labels_all[val_idx]))


print "===> (7) accuracy of HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores), np.std(scores), scores)

