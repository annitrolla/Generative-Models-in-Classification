import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier, MultinomialHMMClassifier
import cPickle
 
# load data
train_static_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_static_numeric.npy')
train_static_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_static_nonnumeric.npy')
train_dynamic_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_dynamic_numeric.npy')
train_dynamic_nonnumeric_pos = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_dynamic_nonnumeric_pos.npy')
train_dynamic_nonnumeric_neg = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_dynamic_nonnumeric_neg.npy')

test_static_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_static_numeric.npy')
test_static_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_static_nonnumeric.npy')
test_dynamic_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_dynamic_numeric.npy')
test_dynamic_nonnumeric_pos = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_dynamic_nonnumeric_pos.npy')
test_dynamic_nonnumeric_neg = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_dynamic_nonnumeric_neg.npy')

train_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_labels.npy')
test_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_labels.npy')

# load mapping
with open('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/pos_to_neg.pkl', 'r') as f:
    pos_to_neg = cPickle.load(f)
with open('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/neg_to_pos.pkl', 'r') as f:
    neg_to_pos = cPickle.load(f)

# GMMHMM
#hmmcl = GMMHMMClassifier()
#model_pos, model_neg = hmmcl.train(3, 10, 10, 'full', train_dynamic_numeric, train_labels)
#print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, test_dynamic_numeric, test_labels)

# MultinomialHMMClassifier
hmmcl = MultinomialHMMClassifier()
models_pos, models_neg = hmmcl.train_per_feature(3, 10, train_dynamic_nonnumeric_pos, train_dynamic_nonnumeric_neg, train_labels)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test_per_feature(model_pos, model_neg, test_dynamic_numeric, 
                                                                                   test_labels, pos_to_neg, neg_to_pos)
