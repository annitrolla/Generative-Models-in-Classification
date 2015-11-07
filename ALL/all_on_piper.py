"""

Generate a dataset with the following properties
    - Binary classification
    - Static features which are ~0.7 separable using RF
    - Dynamic features which are ~0.7 separable using generative-model-based classifier
    - Some of the instances must be predicatable only from the static features, other only from
      the sequence data, other from both, and some are not reasonably predictable at all

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
from LSTM.lstm_classifier import LSTMClassifier 

# general parameters
lstm_nepochs = 20
rf_estimators = 500

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_labels.npy')

dynamic_train[:,[3,4,5,6,7,8,9,10,11,12,13,14],:] = dynamic_train[:,[3,4,5,6,7,8,9,10,11,12,13,14],:]/10000000.0
dynamic_val[:,[3,4,5,6,7,8,9,10,11,12,13,14],:] = dynamic_val[:,[3,4,5,6,7,8,9,10,11,12,13,14],:]/10000000.0

dynamic_train = dynamic_train[:,[3,6,7,8,9,10,11,12,13,14],:]
dynamic_val = dynamic_val[:,[3,6,7,8,9,10,11,12,13,14],:]

# static data with RF
rf = RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1)
rf.fit(static_train, labels_train)
print "Random Forest with static features on validation set: %.4f" % rf.score(static_val, labels_val)

# dynamic data with HMM
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, dynamic_train, labels_train)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, dynamic_val, labels_val)

# dynamic data with RF
print "Training RF on the dynamic dataset..."
dynamic_as_static_train = dynamic_train.reshape((dynamic_train.shape[0], dynamic_train.shape[1] * dynamic_train.shape[2]))
dynamic_as_static_val = dynamic_val.reshape((dynamic_val.shape[0], dynamic_val.shape[1] * dynamic_val.shape[2]))
rf = RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1)
rf.fit(dynamic_as_static_train, labels_train)
print "RF with dynamic features on validation set: %.4f" % rf.score(dynamic_as_static_val, labels_val)

# dynamic data with LSTM
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(dynamic_train, labels_train)
print "LSTM with dynamic features on validation set: %.4f" % lstmcl.test(model_pos, model_neg, dynamic_val, labels_val)

# joint models
print ""
print "Splitting data in two halves..."
fh_idx = np.random.choice(range(0, dynamic_train.shape[0]), size=np.round(dynamic_train.shape[0] * 0.5, 0), replace=False)
sh_idx = list(set(range(0, dynamic_train.shape[0])) - set(fh_idx))
fh_data = dynamic_train[fh_idx, :, :]
fh_labels = labels_train[fh_idx]
sh_data = dynamic_train[sh_idx, :, :]
sh_labels = labels_train[sh_idx]

# RF+HMM
print "Evaluating RF+HMM model:"

print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, fh_data, fh_labels)

print "Extracting ratios based on the HMM model..."
sh_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val)

print "Merging static features and HMM-based ratios..."
enriched_sh_data = np.hstack((static_train[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1)
rf.fit(enriched_sh_data, sh_labels)
print "RF+HMM with enriched features on validation set: %.4f" % rf.score(enriched_val_data, labels_val)

# RF+LSTM
print "Evaluating RF+LSTM model:"

print "Training LSTM classifier..."
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(fh_data, fh_labels)

print "Extracting ratios based on the LSTM model..."
sh_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val)

print "Merging static features and LSTM-based ratios..."
enriched_sh_data = np.hstack((static_train[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1)
rf.fit(enriched_sh_data, sh_labels)
print "RF+LSTM with enriched features on validation set: %.4f" % rf.score(enriched_val_data, labels_val)




