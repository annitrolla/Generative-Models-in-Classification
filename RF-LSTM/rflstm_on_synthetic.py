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


# parameters
nsamples = 10000
nq = nsamples / 4
nfeatures = 10
nseqfeatures = 20
seqlen = 30

s_pos_mu = 0.0
s_pos_sd = 1.0
s_neg_mu = 5.0
s_neg_sd = 1.0


#
# Generate data
#

# generate labels
labels = np.array([1] * nq + [0] * nq + [1] * nq + [0] * nq)

# generate static features
static = np.vstack((np.random.normal(5.0, 1.0, (nq, nfeatures)),
                    np.random.normal(0.0, 1.0, (nq, nfeatures)),
                    np.random.normal(0.0, 1.0, (nq, nfeatures)),
                    np.random.normal(0.0, 1.0, (nq, nfeatures))))

# generate dynamic features
coefs_q1 = np.random.normal(0.2, 1, seqlen - 1)
coefs_q2 = np.random.normal(0.2, 1, seqlen - 1)
coefs_q3 = np.random.normal(0.0, 1, seqlen - 1)
coefs_q4 = np.random.normal(0.2, 1, seqlen - 1)
dynamic_q1 = np.empty((nq, nseqfeatures, seqlen))
dynamic_q2 = np.empty((nq, nseqfeatures, seqlen))
dynamic_q3 = np.empty((nq, nseqfeatures, seqlen))
dynamic_q4 = np.empty((nq, nseqfeatures, seqlen))
for i in range(nq):
    dynamic_q1[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
    dynamic_q2[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
    dynamic_q3[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
    dynamic_q4[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
for s in range(nq):
    for t in range(1, seqlen):
        dynamic_q1[s, :, t] = coefs_q1[t - 1] * dynamic_q1[s, :, t - 1] 
        dynamic_q2[s, :, t] = coefs_q2[t - 1] * dynamic_q2[s, :, t - 1]
        dynamic_q3[s, :, t] = coefs_q3[t - 1] * dynamic_q3[s, :, t - 1]
        dynamic_q4[s, :, t] = coefs_q4[t - 1] * dynamic_q4[s, :, t - 1]
dynamic = np.vstack((dynamic_q1, dynamic_q2, dynamic_q3, dynamic_q4))


#
# Split the dataset into training and validation
#

# pick samples for training and for validation
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
val_idx = list(set(range(0, nsamples)) - set(train_idx))

static_train = static[train_idx, :]
dynamic_train = dynamic[train_idx, :, :]
static_val = static[val_idx, :]
dynamic_val = dynamic[val_idx, :, :]
labels_train = labels[train_idx]
labels_val = labels[val_idx]


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





