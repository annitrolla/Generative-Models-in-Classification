"""

Train the whole set related to HMMs  on the 'inforegister' dataset
 

"""

import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier

# general parameters
nfolds = 5
nestimators = 500
nhmmstates = 3
nhmmiter = 100
hmmcovtype = "full"  # options: full, diag, spherical

#
# Load the dataset
#
print 'Loading the dataset..'
static_all = np.load('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_static.npy')
dynamic_all = np.load('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_dynamic.npy')
labels_all = np.load('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_labels.npy')

nsamples = static_all.shape[0]

#
# k-fold CV
#

# prepare where to store the predictions
predictions_all_rf = np.empty((len(labels_all), 2))
predictions_all_hmm = np.empty((len(labels_all), 2))

# prepare where to store the ratios
ratios_all_hmm = np.empty(len(labels_all))

# split indices into folds
predict_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
for fid, predict_idx in enumerate(predict_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(predict_idx))
    
    # extract predictions using RF on static
    print "    Extracting predictions on static data with RF..."
    print labels_all[train_idx].shape
    
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(static_all[train_idx], labels_all[train_idx])
    predictions_all_rf[predict_idx] = rf.predict_log_proba(static_all[predict_idx])
    predictions_all_rf[predictions_all_rf == -inf] = np.min(predictions_all_rf[predictions_all_rf != -inf])

    # extract predictions using HMM on dynamic
    print "    Extracting predictions on dynamic data with HMM..."
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                       dynamic_all[train_idx], labels_all[train_idx])
    predictions_all_hmm[predict_idx] = hmmcl.predict_log_proba(model_pos, model_neg, dynamic_all[predict_idx])
    ratios_all_hmm[predict_idx] = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[predict_idx])

#
# Prepare combined datasets for the future experiments
#

# datasets for ensemble learning
predictions_combined_rf_hmm = np.concatenate((predictions_all_rf, predictions_all_hmm), axis=1)

# datasets for hybrid learning
enriched_by_hmm = np.concatenate((static_all, np.matrix(ratios_all_hmm).T), axis=1)

# dataset to confirm that RF on dynamic is not better than generative models on dynamic data
dynamic_as_static = dynamic_all.reshape((dynamic_all.shape[0], dynamic_all.shape[1] * dynamic_all.shape[2]))

# dataset to confirm that RF on naive combination of features is not better than our fancy models
static_and_dynamic_as_static = np.concatenate((static_all, dynamic_as_static), axis=1)

# dataset to check how generative models perform if provided with static features along with dynamic
static_as_dynamic = np.zeros((static_all.shape[0], static_all.shape[1], dynamic_all.shape[2]))
for i in range(static_all.shape[0]):
    static_as_dynamic[i, :, :] = np.tile(static_all[i, :], (dynamic_all.shape[2], 1)).T
dynamic_and_static_as_dynamic = np.concatenate((dynamic_all, static_as_dynamic + np.random.uniform(-0.0001, 0.0001, static_as_dynamic.shape)), axis=1)

#
# k-fold CV for performance estimation
#
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # Ensemble on predictions by RF and HMM (1)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(predictions_combined_rf_hmm[train_idx], labels_all[train_idx])
    scores[1].append(rf.score(predictions_combined_rf_hmm[val_idx], labels_all[val_idx]))

    # Hybrid on features enriched by HMM (3)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
    scores[3].append(rf.score(enriched_by_hmm[val_idx], labels_all[val_idx]))

    # RF on static features (5)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(static_all[train_idx], labels_all[train_idx])
    scores[5].append(rf.score(static_all[val_idx], labels_all[val_idx]))

    # RF on dynamic features (6)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(dynamic_as_static[train_idx], labels_all[train_idx])
    scores[6].append(rf.score(dynamic_as_static[val_idx], labels_all[val_idx]))

    # HMM on dynamic features (7)
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, dynamic_all[train_idx], labels_all[train_idx])
    acc, auc = hmmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx])
    scores[7].append(acc)

    # HMM on dynamic and static (turned into fake sequences) (9)
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, dynamic_and_static_as_dynamic[train_idx], labels_all[train_idx])
    scores[9].append(hmmcl.test(model_pos, model_neg, dynamic_and_static_as_dynamic[val_idx], labels_all[val_idx]))

    # RF on static and dynamic (spatialized) features (11)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(static_and_dynamic_as_static[train_idx], labels_all[train_idx])
    scores[11].append(rf.score(static_and_dynamic_as_static[val_idx], labels_all[val_idx]))

print "===> (1) Ensemble (RF) on predictions by RF and HMM: %.4f (+/- %.4f) %s" % (np.mean(scores[1]), np.std(scores[1]), scores[1])
print "===> (3) Hybrid (RF) on features enriched by HMM: %.4f (+/- %.4f) %s" % (np.mean(scores[3]), np.std(scores[3]), scores[3])
print "===> (5) RF on static features: %.4f (+/- %.4f) %s" % (np.mean(scores[5]), np.std(scores[5]), scores[5])
print "===> (6) RF on dynamic (spatialized) features: %.4f (+/- %.4f) %s" % (np.mean(scores[6]), np.std(scores[6]), scores[6])
print "===> (7) HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[7]), np.std(scores[7]), scores[7])
print "===> (9) HMM on dynamic and static features: %.4f (+/- %.4f) %s" % (np.mean(scores[9]), np.std(scores[9]), scores[9])
print "===> (11) RF on static and dynamic (spatialized) features: %.4f (+/- %.4f) %s" % (np.mean(scores[11]), np.std(scores[11]), scores[11])


