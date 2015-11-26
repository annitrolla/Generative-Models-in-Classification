import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier

train_static = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_static.npy' % dataset)
train_dynamic = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_dynamic.npy' % dataset)
test_static = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_static.npy' % dataset)
test_dynamic = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_dynamic.npy' % dataset)
train_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_labels.npy' % dataset)
test_labels = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_labels.npy' % dataset)

