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


#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_multisame/test_labels.npy')

#
# Evaluating Joint Model
#
print "Evaluating joint model:"
print "Splitting data in two halves..."
fh_idx = np.random.choice(range(0, dynamic_train.shape[0]), size=np.round(dynamic_train.shape[0] * 0.5, 0), replace=False)
sh_idx = list(set(range(0, dynamic_train.shape[0])) - set(fh_idx))
fh_data = dynamic_train[fh_idx, :, :]
fh_labels = labels_train[fh_idx]
sh_data = dynamic_train[sh_idx, :, :]
sh_labels = labels_train[sh_idx]

print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, fh_data, fh_labels)
print model_pos.startprob, model_pos.transmat, model_pos.means, model_pos.covars
print model_neg.startprob, model_neg.transmat, model_neg.means, model_neg.covars





