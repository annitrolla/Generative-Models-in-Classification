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
nseqfeatures = 20
seqlen = 30

s_pos_mu = 0.0
s_pos_sd = 1.0
s_neg_mu = 5.0
s_neg_sd = 1.0

error_ratio = 0.3


#
# Generate data
#

# generate labels
pos_labels = [1] * nsamples
neg_labels = [0] * nsamples

# generate static features
static_pos = np.random.normal(s_pos_mu, s_pos_sd, (nsamples, nfeatures))
static_neg = np.random.normal(s_neg_mu, s_neg_sd, (nsamples, nfeatures))
static_data = np.vstack((static_pos, static_neg))
static_labels = np.hstack((pos_labels, neg_labels))

# generate dynamic features
pos_coefs = np.random.normal(0.0, 1, seqlen - 1)
neg_coefs = np.random.normal(0.5, 1, seqlen - 1)
dynamic_pos = np.empty((nsamples, nseqfeatures, seqlen))
dynamic_neg = np.empty((nsamples, nseqfeatures, seqlen))
for i in range(nsamples):
    dynamic_pos[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
    dynamic_neg[i, :, 0] = np.random.uniform(-1.0, 1.0, nseqfeatures)
for s in range(nsamples):
    for t in range(1, seqlen):
        dynamic_pos[s, :, t] = pos_coefs[t - 1] * dynamic_pos[s, :, t - 1] 
        dynamic_neg[s, :, t] = neg_coefs[t - 1] * dynamic_neg[s, :, t - 1]
dynamic_data = np.vstack((dynamic_pos, dynamic_neg))
dynamic_labels = np.hstack((pos_labels, neg_labels))


#
# Split the dataset into training and validation
#

# pick samples for training and for validation
train_idx = np.random.choice(range(0, 2 * nsamples), size=np.round(2 * nsamples * 0.7, 0), replace=False)
val_idx = list(set(range(0, 2 * nsamples)) - set(train_idx))

# split static featureset
static_train_data = static_data[train_idx, :]
static_train_labels = static_labels[train_idx]
static_val_data = static_data[val_idx, :]
static_val_labels = static_labels[val_idx]

# split dynamic featureset
dynamic_train_data = dynamic_data[train_idx, :, :]
dynamic_train_labels = dynamic_labels[train_idx]
dynamic_val_data = dynamic_data[val_idx, :, :]
dynamic_val_labels = dynamic_labels[val_idx]


#
# Introduce errors into both training and validation dataset so that some percentage of samples is misclassified
#

# pick samples in the training set to become misclassifed
train_error_idx = np.random.choice(range(0, static_train_data.shape[0]), size=np.round(static_train_data.shape[0] * (error_ratio * 1.5), 0), replace=False)
misclassified_train_static = train_error_idx[0 : len(train_error_idx) / 3]
misclassified_train_dynamic = train_error_idx[len(train_error_idx) / 3 : 2 * (len(train_error_idx) / 3)]
misclassified_train_both = train_error_idx[2 * (len(train_error_idx) / 3) : ]

# pick samples in the validation set to become misclassified
val_error_idx = np.random.choice(range(0, static_val_data.shape[0]), size=np.round(static_val_data.shape[0] * (error_ratio * 1.5), 0), replace=False)
misclassified_val_static = val_error_idx[0 : len(val_error_idx) / 3]
misclassified_val_dynamic = val_error_idx[len(val_error_idx) / 3 : 2 * (len(val_error_idx) / 3)]
misclassified_val_both = val_error_idx[2 * (len(val_error_idx) / 3) : ]

# flip labels for the training set
static_train_labels[misclassified_train_static] = 0 ** static_train_labels[misclassified_train_static]
dynamic_train_labels[misclassified_train_dynamic] = 0 ** dynamic_train_labels[misclassified_train_dynamic]
static_train_labels[misclassified_train_both] = 0 ** static_train_labels[misclassified_train_both]
dynamic_train_labels[misclassified_train_both] = 0 ** dynamic_train_labels[misclassified_train_both]

# flip labels for the validation set
static_val_labels[misclassified_val_static] = 0 ** static_val_labels[misclassified_val_static]
dynamic_val_labels[misclassified_val_dynamic] = 0 ** dynamic_val_labels[misclassified_val_dynamic]
static_val_labels[misclassified_val_both] = 0 ** static_val_labels[misclassified_val_both]
dynamic_val_labels[misclassified_val_both] = 0 ** dynamic_val_labels[misclassified_val_both]

# check shapes
#print static_train_data.shape
#print static_train_labels.shape
#print static_val_data.shape
#print static_val_labels.shape
#print dynamic_train_data.shape
#print dynamic_train_labels.shape
#print dynamic_val_data.shape
#print dynamic_val_labels.shape


#
# Sanity Checks
#

print "Error ratio: %.2f, expected performance of a lonely model is %.2f, of the joint model %.2f" % (error_ratio, 1 - error_ratio, 1 - error_ratio + error_ratio / 2.0)

# a) static data classification
rf = RandomForestClassifier(n_estimators=100)
rf.fit(static_train_data, static_train_labels)
print "Random Forest with static features on validation set: %.4f" % rf.score(static_val_data, static_val_labels)

# b) dynamic data classification
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, dynamic_train_data, dynamic_train_labels)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, dynamic_val_data, dynamic_val_labels)


#
# Evaluating Joint Model
#

print ""
print "Evaluating joint model:"
print "Splitting data in two halves..."
fh_idx = np.random.choice(range(0, dynamic_train_data.shape[0]), size=np.round(dynamic_train_data.shape[0] * 0.5, 0), replace=False)
sh_idx = list(set(range(0, dynamic_train_data.shape[0])) - set(fh_idx))
fh_data = dynamic_train_data[fh_idx, :, :]
fh_labels = dynamic_train_labels[fh_idx]
sh_data = dynamic_train_data[sh_idx, :, :]
sh_labels = dynamic_train_labels[sh_idx]

print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, fh_data, fh_labels)

print "Extracting ratios based on the HMM model..."
sh_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val_data)

print "Merging static features and HMM-based ratios..."
enriched_sh_data = np.hstack((static_train_data[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val_data, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=100)
rf.fit(enriched_sh_data, sh_labels)

print np.sum(static_val_labels == dynamic_val_labels), len(static_val_labels)

print "RF+HMM with enriched features on validation set: %.4f" % rf.score(enriched_val_data, static_val_labels)





