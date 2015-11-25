import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier, MultinomialHMMClassifier
 

train_static_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_static_numeric.npy')
train_static_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_static_nonnumeric.npy')
train_dynamic_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_dynamic_numeric.npy')
train_dynamic_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_dynamic_nonnumeric.npy')

test_static_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_static_numeric.npy')
test_static_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_static_nonnumeric.npy')
test_dynamic_numeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_dynamic_numeric.npy')
test_dynamic_nonnumeric = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_dynamic_nonnumeric.npy')

train_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/train_labels.npy')
test_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1var/preprocessed/test_labels.npy')


# GMMHMM
#train_dynamic_numeric
#test_dynamic_numeric

hmmcl = GMMHMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, 10, 'full', train_dynamic_numeric, train_labels)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, test_dynamic_numeric, test_labels)



# MultinomialHMMClassifier
#train_dynamic_nonnumeric
#test_dynamic_nonnumeric
hmmcl = MultinomialHMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, train_dynamic_nonnumeric, train_labels)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, test_dynamic_numeric, test_labels)
