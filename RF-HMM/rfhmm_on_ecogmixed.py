"""

The experiment combines the Fourier features and the features extraced from HMM generative model.
After this augmented dataset is created we train and evaluate Random Forest on it.

"""

import numpy as np
from HMM.hmm_classifier import HMMClassifier 
from sklearn.ensemble import RandomForestClassifier

# parameters
nhmmstates = 3
nestimators = 500
nhmmiter = 20
nfolds = 5
hmmcovtype = 'full'

# load the dataset
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')
    
# merge train and test
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]

# prepare where to store the ratios
ratios_all_hmm = np.empty(len(labels_all))

# split indices into folds
enrich_idx_list = np.array_split(range(nsamples), nfolds)

# CV for dataset enrichment
for fid, enrich_idx in enumerate(enrich_idx_list):
    print "Current fold is %d / %d" % (fid, nfolds)
    train_idx = list(set(range(nsamples)) - set(enrich_idx))

    # extract predictions using HMM on dynamic
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                       dynamic_all[train_idx], labels_all[train_idx])
    ratios_all_hmm[enrich_idx] = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[enrich_idx])
    
# dataset for hybrid learning
enriched_by_hmm = np.concatenate((static_all, np.matrix(ratios_all_hmm).T), axis=1)
    
# CV for accuracy estimation
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    # Hybrid on features enriched by HMM (3)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
    scores.append(rf.score(enriched_by_hmm[val_idx], labels_all[val_idx]))
    
print "===> (3) Hybrid (RF) on features enriched by HMM: %.4f (+/- %.4f) %s" % (mean(scores), std(scores), scores)

