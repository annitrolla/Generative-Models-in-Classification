"""

Try different LSTM architectures and different enrichment schemes on synthetic data
  - Generative LSTM, enrichment with ratios
  - Generative LSTM, enrichment with LSTM layer activations
  - Discriminative LSTM, enrichment with ratios
  - Discriminative LSTM, enrichment with LSTM layer activations

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier 


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
ratios_generative = np.empty((len(labels_all), 2))
ratios_discriminative = np.empty((len(labels_all), 2))

TO BE CONTINUED

predict_idx_list = np.array_split(range(nsamples), nfolds)
for fid, predict_idx in enumerate(predict_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(predict_idx))
    
    # extract visit counts from HMM on dynamic
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nstates, niter, covtype, dynamic_all[train_idx], labels_all[train_idx])
    visits_pos, visits_neg = hmmcl.state_visits(model_pos, model_neg, dynamic_all[predict_idx])
    visits_all_hmm[predict_idx] = np.hstack((visits_pos, visits_neg))

#
# Sanity Checks
#
print "Expected performance of a lonely model is 0.75, of the joint model 1.0"

# static data with RF
rf = RandomForestClassifier(n_estimators=100)
rf.fit(static_train, labels_train)
print "Random Forest with static features on validation set: %.4f" % rf.score(static_val, labels_val)

# dynamic data with HMM
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, dynamic_train, labels_train)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, dynamic_val, labels_val)

# dynamic data with LSTM
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', 20)
model_pos, model_neg = lstmcl.train(dynamic_train, labels_train)
print "LSTM with dynamic features on validation set: %.4f" % lstmcl.test(model_pos, model_neg, dynamic_val, labels_val)

# dynamic data with RF
print "Training RF on the dynamic dataset..."
dynamic_as_static_train = dynamic_train.reshape((dynamic_train.shape[0], dynamic_train.shape[1] * dynamic_train.shape[2]))
dynamic_as_static_val = dynamic_val.reshape((dynamic_val.shape[0], dynamic_val.shape[1] * dynamic_val.shape[2]))
rf = RandomForestClassifier(n_estimators=100)
rf.fit(dynamic_as_static_train, labels_train)
print "RF with dynamic features on validation set: %.4f" % rf.score(dynamic_as_static_val, labels_val)


#
# Evaluating Joint Model
#
print ""
print "Evaluating joint (RF+LSTM) model:"
print "Splitting data in two halves..."
fh_idx = np.random.choice(range(0, dynamic_train.shape[0]), size=np.round(dynamic_train.shape[0] * 0.5, 0), replace=False)
sh_idx = list(set(range(0, dynamic_train.shape[0])) - set(fh_idx))
fh_data = dynamic_train[fh_idx, :, :]
fh_labels = labels_train[fh_idx]
sh_data = dynamic_train[sh_idx, :, :]
sh_labels = labels_train[sh_idx]

print "Training LSTM classifier..."
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', 20)
model_pos, model_neg = lstmcl.train(fh_data, fh_labels)

print "Extracting ratios based on the LSTM model..."
sh_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val)

print "Merging static features and LSTM-based ratios..."
enriched_sh_data = np.hstack((static_train[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=100)
rf.fit(enriched_sh_data, sh_labels)
print "RF+LSTM with enriched features on validation set: %.4f" % rf.score(enriched_val_data, labels_val)




