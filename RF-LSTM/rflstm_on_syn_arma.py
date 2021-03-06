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
from LSTM.lstm_classifier import LSTMClassifier 

#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_arma/test_labels.npy')


#
# Sanity Checks
#
print "Expected performance of a lonely model is 0.75, of the joint model 1.0"

# a) static data classification
rf = RandomForestClassifier(n_estimators=100)
rf.fit(static_train, labels_train)
print "Random Forest with static features on validation set: %.4f" % rf.score(static_val, labels_val)

# b) dynamic data classification
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', 20)
model_pos, model_neg = lstmcl.train(dynamic_train, labels_train)
print "LSTM with dynamic features on validation set: %.4f" % lstmcl.test(model_pos, model_neg, dynamic_val, labels_val)


#
# Evaluating Joint Model
#
print ""
print "Evaluating joint model:"
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





