"""

Explore all models' performance for the following parameters:
nsamples, nfeatures, nseqfeatures, seqlen

"""

import numpy as np
from numpy import inf
import argparse
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
from LSTM.lstm_classifier import LSTMClassifier 
from DataNexus.gensyn_lstm_wins import generate_lstm_wins

# read command line arguments
parser = argparse.ArgumentParser(description='Train all models for a given set of parameters.')
parser.add_argument('-n', '--nsamples', dest='nsamples', type=int, required=True,
                          help='total number of samples in the generated dataset')
parser.add_argument('-s', '--nfeatures', dest='nfeatures', type=int, required=True,
                          help='number of static features')
parser.add_argument('-d', '--nseqfeatures', dest='nseqfeatures', type=int, required=True,
                          help='number of dynamic features')
parser.add_argument('-l', '--seqlen', dest='seqlen', type=int, required=True,
                          help='length of a generated sequence')
args = parser.parse_args()

#
# Parameters
#

# read parameters
nsamples = int(args.nsamples)
nfeatures = int(args.nfeatures)
nseqfeatures = int(args.nseqfeatures)
seqlen = int(args.seqlen)

# general parameters
nestimators = 100
nhmmstates = 2
nhmmiter = 10
hmmcovtype = "full" 
lstmsize = 256
lstmdropout = 0.0
lstmoptim = 'rmsprop'
lstmnepochs = 20
lstmbatchsize = 32

# open file to store results
f = open('../../Results/grid_lstm_wins.csv', 'a')


#
# Load data
#

# generate the dataset
train_static, train_dynamic, test_static, test_dynamic, train_labels, test_labels = generate_lstm_wins(nsamples, nfeatures, nseqfeatures, seqlen) 
train_nsamples = train_static.shape[0]
test_nsamples = test_static.shape[0]

# split training into two halves
train_half = train_nsamples / 2
trainA_static = train_static[:train_half]
trainB_static = train_static[train_half:]
trainA_dynamic = train_dynamic[:train_half]
trainB_dynamic = train_dynamic[train_half:]
trainA_labels = train_labels[:train_half]
trainB_labels = train_labels[train_half:]


#
# Train enrichment models
#

# extract predictions using RF on static
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainA_static, trainA_labels)
predictions_trainB_rf = rf.predict_log_proba(trainB_static)
predictions_trainB_rf[predictions_trainB_rf == -inf] = np.min(predictions_trainB_rf[predictions_trainB_rf != -inf])
predictions_test_rf = rf.predict_log_proba(test_static)
predictions_test_rf[predictions_test_rf == -inf] = np.min(predictions_test_rf[predictions_test_rf != -inf])

# extract predictions using HMM on dynamic
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, trainA_dynamic, trainA_labels)
predictions_trainB_hmm = hmmcl.predict_log_proba(model_pos, model_neg, trainB_dynamic)
ratios_trainB_hmm = hmmcl.pos_neg_ratios(model_pos, model_neg, trainB_dynamic)
predictions_test_hmm = hmmcl.predict_log_proba(model_pos, model_neg, test_dynamic)
ratios_test_hmm = hmmcl.pos_neg_ratios(model_pos, model_neg, test_dynamic)

# extract predictions using LSTM on dynamic
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainA_dynamic, trainA_labels)

mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, trainB_dynamic)
predictions_trainB_lstm = np.vstack((mse_pos, mse_neg)).T
ratios_trainB_lstm = lstmcl.pos_neg_ratios(model_pos, model_neg, trainB_dynamic)
trainB_activations_pos = lstmcl.activations(model_pos, trainB_dynamic)
trainB_activations_neg = lstmcl.activations(model_neg, trainB_dynamic)
trainB_activations = np.concatenate((trainB_activations_pos[:, -1, :], trainB_activations_neg[:, -1, :]), axis=1)

mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, test_dynamic)
predictions_test_lstm = np.vstack((mse_pos, mse_neg)).T
ratios_test_lstm = lstmcl.pos_neg_ratios(model_pos, model_neg, test_dynamic)
test_activations_pos = lstmcl.activations(model_pos, test_dynamic)
test_activations_neg = lstmcl.activations(model_neg, test_dynamic)
test_activations = np.concatenate((test_activations_pos[:, -1, :], test_activations_neg[:, -1, :]), axis=1)


#
# Combine datasets
#

# datasets for ensemble learning
trainB_predictions_combined_rf_hmm = np.concatenate((predictions_trainB_rf, ratios_trainB_hmm.reshape((ratios_trainB_hmm.shape[0], 1))), axis=1)
test_predictions_combined_rf_hmm = np.concatenate((predictions_test_rf, ratios_test_hmm.reshape((ratios_test_hmm.shape[0], 1))), axis=1)
trainB_predictions_combined_rf_lstm = np.concatenate((predictions_trainB_rf, ratios_trainB_lstm.reshape((ratios_trainB_lstm.shape[0], 1))), axis=1)
test_predictions_combined_rf_lstm = np.concatenate((predictions_test_rf, ratios_test_lstm.reshape((ratios_test_lstm.shape[0], 1))), axis=1)

# datasets for hybrid learning
trainB_enriched_by_hmm = np.concatenate((trainB_static, np.matrix(ratios_trainB_hmm).T), axis=1)
test_enriched_by_hmm = np.concatenate((test_static, np.matrix(ratios_test_hmm).T), axis=1)
trainB_enriched_by_lstm = np.concatenate((trainB_static, np.matrix(ratios_trainB_lstm).T), axis=1)
test_enriched_by_lstm = np.concatenate((test_static, np.matrix(ratios_test_lstm).T), axis=1)
trainB_activations_by_lstm = np.concatenate((trainB_static, trainB_activations), axis=1)
test_activations_by_lstm = np.concatenate((test_static, test_activations), axis=1)

# dataset to confirm that RF on dynamic is not better than generative models on dynamic data
trainB_dynamic_as_static = trainB_dynamic.reshape((trainB_dynamic.shape[0], trainB_dynamic.shape[1] * trainB_dynamic.shape[2]))
test_dynamic_as_static = test_dynamic.reshape((test_dynamic.shape[0], test_dynamic.shape[1] * test_dynamic.shape[2]))

# dataset to confirm that RF on naive combination of features is not better than our fancy models
trainB_static_and_dynamic_as_static = np.concatenate((trainB_static, trainB_dynamic_as_static), axis=1)
test_static_and_dynamic_as_static = np.concatenate((test_static, test_dynamic_as_static), axis=1)

# dataset to check how generative models perform if provided with static features along with dynamic
trainB_static_as_dynamic = np.zeros((trainB_static.shape[0], trainB_static.shape[1], trainB_dynamic.shape[2]))
for i in range(trainB_static.shape[0]):
    trainB_static_as_dynamic[i, :, :] = np.tile(trainB_static[i, :], (trainB_dynamic.shape[2], 1)).T
trainB_dynamic_and_static_as_dynamic = np.concatenate((trainB_dynamic, trainB_static_as_dynamic + np.random.uniform(-0.0001, 0.0001, trainB_static_as_dynamic.shape)), axis=1)

test_static_as_dynamic = np.zeros((test_static.shape[0], test_static.shape[1], test_dynamic.shape[2]))
for i in range(test_static.shape[0]):
    test_static_as_dynamic[i, :, :] = np.tile(test_static[i, :], (test_dynamic.shape[2], 1)).T
test_dynamic_and_static_as_dynamic = np.concatenate((test_dynamic, test_static_as_dynamic + np.random.uniform(-0.0001, 0.0001, test_static_as_dynamic.shape)), axis=1)


#
# Evaluate 12 models
#

# start storing results
result = "\n%d, %d, %d, %d" % (nsamples, nfeatures, nseqfeatures, seqlen)
f.write(result)
f.flush()
result = ""
scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0}

# Ensemble on predictions by RF and HMM (1)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_predictions_combined_rf_hmm, trainB_labels)
scores[1] = rf.score(test_predictions_combined_rf_hmm, test_labels)
    
# Ensemble on predictions by RF and LSTM (2)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_predictions_combined_rf_lstm, trainB_labels)
scores[2] = rf.score(test_predictions_combined_rf_lstm, test_labels)

# Hybrid on features enriched by HMM (3)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_enriched_by_hmm, trainB_labels)
scores[3] = rf.score(test_enriched_by_hmm, test_labels)

# Hybrid on features enriched by LSTM (4)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_enriched_by_lstm, trainB_labels)
scores[4] = rf.score(test_enriched_by_lstm, test_labels)

# RF on static features (5)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_static, trainB_labels)
scores[5] = rf.score(test_static, test_labels)

# RF on dynamic features (6)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_dynamic_as_static, trainB_labels)
scores[6] = rf.score(test_dynamic_as_static, test_labels)

# HMM on dynamic features (7)
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, trainB_dynamic, trainB_labels)
acc, auc = hmmcl.test(model_pos, model_neg, test_dynamic, test_labels)
scores[7] = acc

# LSTM on dynamic features (8)
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainB_dynamic, trainB_labels)
scores[8] = lstmcl.test(model_pos, model_neg, test_dynamic, test_labels)

# HMM on dynamic and static (turned into fake sequences) (9)
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, trainB_dynamic_and_static_as_dynamic, trainB_labels)
acc, auc = hmmcl.test(model_pos, model_neg, test_dynamic_and_static_as_dynamic, test_labels)
scores[9] = acc

# LSTM on dynamic and static (turned into fake sequences) (10)
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainB_dynamic_and_static_as_dynamic, trainB_labels)
scores[10] = lstmcl.test(model_pos, model_neg, test_dynamic_and_static_as_dynamic, test_labels)

# RF on static and dynamic (spatialized) features (11)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_static_and_dynamic_as_static, trainB_labels)
scores[11] = rf.score(test_static_and_dynamic_as_static, test_labels)

# Hybrid on static features and LSTM activations (12)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_activations_by_lstm, trainB_labels)
scores[12] = rf.score(test_activations_by_lstm, test_labels)

for nr in scores:
    result += ", %.4f" % scores[nr]
    
f.write(result)
f.close()

