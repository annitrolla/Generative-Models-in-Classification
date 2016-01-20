"""

Train the whole set of test on the "lstm wins" synthetic dataset

"""

import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
from LSTM.lstm_classifier import LSTMClassifier 
from DataNexus.gensyn_lstm_wins import generate_lstm_wins

# general parameters
nfolds = 2 #5
nestimators = 300
nhmmstates = 3
nhmmiter = 2 #10
hmmcovtype = "full"  # options: full, diag, spherical
lstmsize = 100 #1000
lstmdropout = 0.5
lstmoptim = 'adadelta'
lstmnepochs = 2 #20
lstmbatchsize = 64


#
# Generate the dataset and run analysis for each parameter combination of interest
#

# tuples in format (#static, #dynamic)
params = [(10000, 10), (1000, 10), (100, 10), (10, 10), (10, 100), (10, 1000), (10, 10000)]
for p in params:
    n_static = p[0]
    n_dynamic = p[1]
    print "Running with parameters n_static = %d, n_dynamic = %d" % (n_static, n_dynamic)
    static_train, dynamic_train, static_val, dynamic_val, labels_train, labels_val = generate_lstm_wins(5000, n_static, n_dynamic, 70)

    # merge train and test
    static_all = np.concatenate((static_train, static_val), axis=0)
    dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
    labels_all = np.concatenate((labels_train, labels_val), axis=0)
    nsamples = static_all.shape[0]


    # k-fold CV for enrichment

    # prepare where to store the ratios
    ratios_all_hmm = np.empty(len(labels_all))
    ratios_all_lstm = np.empty(len(labels_all))

    # split indices into folds
    predict_idx_list = np.array_split(range(nsamples), nfolds)

    # run CV
    for fid, predict_idx in enumerate(predict_idx_list):
        print "Enrichment fold %d / %d" % (fid + 1, nfolds)
        train_idx = list(set(range(nsamples)) - set(predict_idx))
    
        # extract predictions using HMM on dynamic
        hmmcl = HMMClassifier()
        model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                           dynamic_all[train_idx], labels_all[train_idx])
        ratios_all_hmm[predict_idx] = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[predict_idx])

        # extract predictions using LSTM on dynamic
        lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
        model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
        mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, dynamic_all[predict_idx]) 
        ratios_all_lstm[predict_idx] = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[predict_idx])

    # datasets for hybrid learning
    enriched_by_hmm = np.concatenate((static_all, np.matrix(ratios_all_hmm).T), axis=1)
    enriched_by_lstm = np.concatenate((static_all, np.matrix(ratios_all_lstm).T), axis=1)

    # k-fold CV for performance estimation
    val_idx_list = np.array_split(range(nsamples), nfolds)
    scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
    for fid, val_idx in enumerate(val_idx_list):
        print "Performance fold %d / %d" % (fid + 1, nfolds)
        train_idx = list(set(range(nsamples)) - set(val_idx))

        # Hybrid on features enriched by HMM (3)
        rf = RandomForestClassifier(n_estimators=nestimators)
        rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
        scores[3].append(rf.score(enriched_by_hmm[val_idx], labels_all[val_idx]))
        print "%d, %d, HMM, fold %d importances %s" (n_static, n_dynamic, fid + 1, rf.feature_importances_)

        # Hybrid on features enriched by LSTM (4)
        rf = RandomForestClassifier(n_estimators=nestimators)
        rf.fit(enriched_by_lstm[train_idx], labels_all[train_idx])
        scores[4].append(rf.score(enriched_by_lstm[val_idx], labels_all[val_idx]))
        print "%d, %d, LSTM, fold %d importances %s" (n_static, n_dynamic, fid + 1, rf.feature_importances_)

    print "===> (3) Hybrid (RF) on features enriched by HMM: %.4f (+/- %.4f) %s" % (np.mean(scores[3]), np.std(scores[3]), scores[3])
    print "===> (4) Hybrid (RF) on features enriched by LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores[4]), np.std(scores[4]), scores[4])
