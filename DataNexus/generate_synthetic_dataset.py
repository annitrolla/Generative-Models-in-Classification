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


# parameters
nsamples = 1000
nfeatures = 10
nseqfeatures = 10
seqlen = 30

s_pos_mu = 0.0
s_pos_sd = 1.0
s_neg_mu = 5.0
s_neg_sd = 1.0
s_error_ratio = 0.3

# generate static features
correct_samples = np.random.normal(s_pos_mu, s_pos_sd, (nsamples * (1.0 - s_error_ratio), nfeatures))
wrong_samples = np.random.normal(s_neg_mu, s_neg_sd, (nsamples * s_error_ratio, nfeatures))
static_pos = np.vstack((correct_samples, wrong_samples))
pos_labels = [1] * nsamples

correct_samples = np.random.normal(s_neg_mu, s_neg_sd, (nsamples * (1.0 - s_error_ratio), nfeatures))
wrong_samples = np.random.normal(s_pos_mu, s_pos_sd, (nsamples * s_error_ratio, nfeatures))
static_neg = np.vstack((correct_samples, wrong_samples))
neg_labels = [0] * nsamples

# generate dynamic features
pos_coefs = np.random.uniform(-1.0, 1.0, seqlen - 1)
neg_coefs = np.random.uniform(-2.0, 2.0, seqlen - 1)
dynamic_pos = np.empty((nsamples, nfeatures, seqlen))
dynamic_neg = np.empty((nsamples, nfeatures, seqlen))
for i in range(nsamples):
    dynamic_pos[i, :, 0] = np.random.uniform(-1.0, 1.0, nfeatures)
    dynamic_neg[i, :, 0] = np.random.uniform(-1.0, 1.0, nfeatures)
for s in range(nsamples):
    for t in range(1, seqlen):
        dynamic_pos[s, :, t] = pos_coefs[t - 1] * dynamic_pos[s, :, t - 1] 
        dynamic_neg[s, :, t] = neg_coefs[t - 1] * dynamic_neg[s, :, t - 1]

# sanity check: classification
data = np.vstack((dynamic_pos, dynamic_neg))
labels = np.hstack((pos_labels, neg_labels))
train_idx = np.random.choice(range(0, 2 * nsamples), size=np.round(2 * nsamples * 0.7, 0), replace=False)
val_idx = list(set(range(0, 2 * nsamples)) - set(train_idx))
train_data = data[train_idx, :]
train_labels = labels[train_idx]
val_data = data[val_idx, :]
val_labels = labels[val_idx]

hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(10, 3, train_data, train_labels)
print hmmcl.test(model_pos, model_neg, val_data, val_labels)


"""
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_data, train_labels)

print rf.score(train_data, train_labels)
print rf.score(val_data, val_labels)
"""




