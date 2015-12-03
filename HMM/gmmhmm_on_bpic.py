import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier

train_static = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
train_dynamic = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
test_static = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
test_dynamic = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
train_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
test_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')

hmmcl = GMMHMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, 5, 'full', train_dynamic, train_labels)
print "HMM with dynamic features on validation set: %.4f" % hmmcl.test(model_pos, model_neg, test_dynamic, test_labels)
