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
import statsmodels.tsa.arima_process as ap

# parameters
nsamples = 1000
nq = nsamples / 4
nfeatures = 10
nseqfeatures = 20
seqlen = 1000

arparams_pos = np.array([1, 0.75, 0.4])
maparams_pos =  np.array([1, 0.25, 0.2])
arparams_neg = np.array([1, -0.3, 0.55])
maparams_neg =  np.array([1, 0.65, 0.8])

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
dynamic_q1 = np.vstack([ap.arma_generate_sample(arparams_pos, maparams_pos, seqlen) for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen) 
dynamic_q2 = np.vstack([ap.arma_generate_sample(arparams_pos, maparams_pos, seqlen) for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
dynamic_q3 = np.vstack([ap.arma_generate_sample(arparams_neg, maparams_neg, seqlen) for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
dynamic_q4 = np.vstack([ap.arma_generate_sample(arparams_pos, maparams_pos, seqlen) for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
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
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/train_labels.npy', train_labels)
np.save('/storage/hpc_anna/GMiC/Data/syn_arma/test_labels.npy', test_labels)

print 'Done.'


