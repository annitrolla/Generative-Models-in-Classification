"""

Random Forest on static features

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#
# Load the dataset
#
print 'Loading the dataset...'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')

# static data with RF
print 'Training RF...'
rf = RandomForestClassifier(n_estimators=100)
rf.fit(static_train, labels_train)
print "Random Forest with static features on validation set: %.4f" % rf.score(static_val, labels_val)
