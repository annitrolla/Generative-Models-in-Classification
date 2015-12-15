import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier

# parameters
lstmsize = 500
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 50
lstmbatchsize = 256
nestimators = 500
nfolds = 5

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')

# merge train and test
print "Combining data from train and test for CV predictions..."
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]

# prepare where to store the ratios
activations_all = np.empty((len(labels_all), lstmsize * 2))

# split indices into folds
enrich_idx_list = np.array_split(range(nsamples), nfolds)

# CV for feature enrichment
for fid, enrich_idx in enumerate(enrich_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(enrich_idx))

    # train models for enrichment
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])

    # extract activations
    activations_pos = lstmcl.activations(model_pos, dynamic_all[enrich_idx])
    activations_neg = lstmcl.activations(model_neg, dynamic_all[enrich_idx])
    activations_all[enrich_idx] = np.concatenate((activations_pos[:, -1, :], activations_neg[:, -1, :]), axis=1)

# dataset for hybrid learning
enriched_by_lstm = np.concatenate((static_all, activations_all), axis=1)
print static_all.shape
print enriched_by_lstm.shape

# CV for accuracy estimation
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))
    
    # Hybrid on features enriched by HMM (3)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_lstm[train_idx], labels_all[train_idx])
    scores.append(rf.score(enriched_by_lstm[val_idx], labels_all[val_idx]))

print "===> (4) Hybrid (RF) on features (activations) enriched by LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores), np.std(scores), scores)

