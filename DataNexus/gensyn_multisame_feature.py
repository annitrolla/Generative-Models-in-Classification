"""

Generate a dataset with the following properties
    - Binary classification
    - Static features which are ~0.7 separable using RF
    - Dynamic features which are ~0.7 separable using generative-model-based classifier
    - Some of the instances are predicatable only from the static features, other only from
      the sequence data, other from both, and some are not reasonably predictable at all
    - All features are generated from the SAME distribution

"""

import numpy as np


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
print 'Generating data...'

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
# Split the dataset into training and test sets
#
print 'Splitting train and test...'

# pick samples for training and for testing
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

train_static = static[train_idx, :]
train_dynamic = dynamic[train_idx, :, :]
test_static = static[test_idx, :]
test_dynamic = dynamic[test_idx, :, :]
train_labels = labels[train_idx]
test_labels = labels[test_idx]


#
# Store the dataset
#
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/train_labels.npy', train_labels)
np.save('/storage/hpc_anna/GMiC/Data/syn_multisame/test_labels.npy', test_labels)

print 'Done.'


