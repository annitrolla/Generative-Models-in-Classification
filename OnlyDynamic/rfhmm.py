import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
    
# parameters                               
nhmmstates = 3
nestimators = 500
nhmmiter = 10
nfolds = 5
hmmcovtype = 'full'

# Load the dataset
#dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
#dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
#labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
#labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')

dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')

# Merge train and test
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = dynamic_all.shape[0]

# prepare where to store the ratios
ratios_all_hmm = np.empty(len(labels_all))
predictions_all_hmm = np.empty((len(labels_all), 2))
predictions_all = np.empty((len(labels_all), ))

# split indices into folds
enrich_idx_list = np.array_split(range(nsamples), nfolds)

# run CV for enrichment
for fid, enrich_idx in enumerate(enrich_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(enrich_idx))

    # extract predictions using HMM on dynamic
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                       dynamic_all[train_idx], labels_all[train_idx])
    ratios_all_hmm[enrich_idx] = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[enrich_idx])
    predictions_all_hmm[enrich_idx] = hmmcl.predict_log_proba(model_pos, model_neg, dynamic_all[enrich_idx])
    predictions_all[enrich_idx] = hmmcl.predict(hmmcl.tensor_to_list(dynamic_all[enrich_idx]), model_pos, model_neg)
 
# dataset for hybrid learning
dynamic_as_static = dynamic_all.reshape((dynamic_all.shape[0], dynamic_all.shape[1] * dynamic_all.shape[2]))
enriched_by_hmm = np.concatenate((dynamic_as_static, predictions_all_hmm), axis=1)


# k-fold cross validation to obtain accuracy
print '===> HMM on dynamic: %.4f' % hmmcl.accuracy(predictions_all, labels_all)

val_idx_list = np.array_split(range(nsamples), nfolds)
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(dynamic_as_static[train_idx], labels_all[train_idx])
    scores.append(rf.score(dynamic_as_static[val_idx], labels_all[val_idx]))

print '===> RF on dynamic without enrichment: %.4f (+- %.4f) %s' % (np.mean(scores), np.std(scores), scores)


val_idx_list = np.array_split(range(nsamples), nfolds)
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_hmm[train_idx], labels_all[train_idx])
    scores.append(rf.score(enriched_by_hmm[val_idx], labels_all[val_idx]))
   
print '===> RF on dynamic enriched by HMM: %.4f (+- %.4f) %s' % (np.mean(scores), np.std(scores), scores)





