"""

Generate HMM-based classifier on syn_lstm_wins synthetic dataset

"""

import numpy as np
from HMM.hmm_classifier import HMMClassifier

print 'Loading the dataset..'
train_data = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy")

print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, train_data, train_labels)
print hmmcl.test(model_pos, model_neg, test_data, test_labels)





