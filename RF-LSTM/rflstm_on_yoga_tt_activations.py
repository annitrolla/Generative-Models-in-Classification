import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
from LSTM.lstm_classifier import LSTMClassifier 

#
# General parameters
#
nfolds = 5
nestimators = 500
lstmsize = 512
lstmdropout = 0.0
lstmoptim = 'rmsprop'
lstmnepochs = 20
lstmbatchsize = 1


#
# Load the dataset
#
print 'Loading the dataset..'

train_static = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_static.npy')
train_dynamic = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_dynamic.npy')
train_labels = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_labels.npy')
train_nsamples = train_static.shape[0]

test_static = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_static.npy')
test_dynamic = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_dynamic.npy')
test_labels = np.load('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_labels.npy')
test_nsamples = test_static.shape[0]

seqlen = train_dynamic.shape[2]

#
# Split training
#
train_half = train_nsamples / 2
trainA_static = train_static[:train_half]
trainB_static = train_static[train_half:]
trainA_dynamic = train_dynamic[:train_half]
trainB_dynamic = train_dynamic[train_half:]
trainA_labels = train_labels[:train_half]
trainB_labels = train_labels[train_half:]


#
# Train enrichment models on trainA
#
print 'Training enrichment models...'

# train LSTM activations extractor
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainA_dynamic, trainA_labels)

trainB_activations_pos = lstmcl.activations(model_pos, trainB_dynamic)
trainB_activations_neg = lstmcl.activations(model_neg, trainB_dynamic)
trainB_activations = np.concatenate((trainB_activations_pos[:, seqlen-1, :], trainB_activations_neg[:, seqlen-1, :]), axis=1)

test_activations_pos = lstmcl.activations(model_pos, test_dynamic)
test_activations_neg = lstmcl.activations(model_neg, test_dynamic)
test_activations = np.concatenate((test_activations_pos[:, seqlen-1, :], test_activations_neg[:, seqlen-1, :]), axis=1)

#
# Prepare combined datasets for the future experiments
#

# datasets for hybrid learning
trainB_enriched_by_lstm = np.concatenate((trainB_static, trainB_activations), axis=1)
test_enriched_by_lstm = np.concatenate((test_static, test_activations), axis=1)


#
# Training models on trainB and performance estimation on test
#
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}

# Hybrid on static features and LSTM activations (12)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_enriched_by_lstm, trainB_labels)
scores[12].append(rf.score(test_enriched_by_lstm, test_labels))


#
# Print out the results
#
print "===> (12) Hybrid (RF) on features + LSTM activations: %.4f (+/- %.4f) %s" % (np.mean(scores[12]), np.std(scores[12]), scores[12])

