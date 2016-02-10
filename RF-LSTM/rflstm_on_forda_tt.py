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

train_static = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_static.npy')
train_dynamic = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_dynamic.npy')
train_labels = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_labels.npy')
train_nsamples = train_static.shape[0]

test_static = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/test_static.npy')
test_dynamic = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/test_dynamic.npy')
test_labels = np.load('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/test_labels.npy')
test_nsamples = test_static.shape[0]


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

# extract predictions using RF on static
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainA_static, trainA_labels)
predictions_trainB_rf = rf.predict_log_proba(trainB_static)
predictions_trainB_rf[predictions_trainB_rf == -inf] = np.min(predictions_trainB_rf[predictions_trainB_rf != -inf])
predictions_test_rf = rf.predict_log_proba(test_static)
predictions_test_rf[predictions_test_rf == -inf] = np.min(predictions_test_rf[predictions_test_rf != -inf])

# extract predictions using LSTM on dynamic
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainA_dynamic, trainA_labels)
mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, trainB_dynamic)
predictions_trainB_lstm = np.vstack((mse_pos, mse_neg)).T
ratios_trainB_lstm = lstmcl.pos_neg_ratios(model_pos, model_neg, trainB_dynamic)
mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, test_dynamic)
predictions_test_lstm = np.vstack((mse_pos, mse_neg)).T
ratios_test_lstm = lstmcl.pos_neg_ratios(model_pos, model_neg, test_dynamic)


#
# Prepare combined datasets for the future experiments
#

# datasets for ensemble learning
trainB_predictions_combined_rf_lstm = np.concatenate((predictions_trainB_rf, ratios_trainB_lstm.reshape((ratios_trainB_lstm.shape[0], 1))), axis=1)
test_predictions_combined_rf_lstm = np.concatenate((predictions_test_rf, ratios_test_lstm.reshape((ratios_test_lstm.shape[0], 1))), axis=1)

# datasets for hybrid learning
trainB_enriched_by_lstm = np.concatenate((trainB_static, np.matrix(ratios_trainB_lstm).T), axis=1)
test_enriched_by_lstm = np.concatenate((test_static, np.matrix(ratios_test_lstm).T), axis=1)

# dataset to confirm that RF on dynamic is not better than generative models on dynamic data
trainB_dynamic_as_static = trainB_dynamic.reshape((trainB_dynamic.shape[0], trainB_dynamic.shape[1] * trainB_dynamic.shape[2]))
test_dynamic_as_static = test_dynamic.reshape((test_dynamic.shape[0], test_dynamic.shape[1] * test_dynamic.shape[2]))

# dataset to confirm that RF on naive combination of features is not better than our fancy models
trainB_static_and_dynamic_as_static = np.concatenate((trainB_static, trainB_dynamic_as_static), axis=1)
test_static_and_dynamic_as_static = np.concatenate((test_static, test_dynamic_as_static), axis=1)

# dataset to check how generative models perform if provided with static features along with dynamic
#trainB_static_as_dynamic = np.zeros((trainB_static.shape[0], trainB_static.shape[1], trainB_dynamic.shape[2]))
#for i in range(trainB_static.shape[0]):
#    trainB_static_as_dynamic[i, :, :] = np.tile(trainB_static[i, :], (trainB_dynamic.shape[2], 1)).T
#trainB_dynamic_and_static_as_dynamic = np.concatenate((trainB_dynamic, trainB_static_as_dynamic + np.random.uniform(-0.0001, 0.0001, trainB_static_as_dynamic.shape)), axis=1)

#test_static_as_dynamic = np.zeros((test_static.shape[0], test_static.shape[1], test_dynamic.shape[2]))
#for i in range(test_static.shape[0]):
#    test_static_as_dynamic[i, :, :] = np.tile(test_static[i, :], (test_dynamic.shape[2], 1)).T
#test_dynamic_and_static_as_dynamic = np.concatenate((test_dynamic, test_static_as_dynamic + np.random.uniform(-0.0001, 0.0001, test_static_as_dynamic.shape)), axis=1)


#
# Training models on trainB and performance estimation on test
#
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}

# Ensemble on predictions by RF and LSTM (2)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_predictions_combined_rf_lstm, trainB_labels)
scores[2].append(rf.score(test_predictions_combined_rf_lstm, test_labels))

# Hybrid on features enriched by LSTM (4)
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(trainB_enriched_by_lstm, trainB_labels)
scores[4].append(rf.score(test_enriched_by_lstm, test_labels))

# LSTM on dynamic features (8)
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(trainB_dynamic, trainB_labels)
scores[8].append(lstmcl.test(model_pos, model_neg, test_dynamic, test_labels))

# LSTM on dynamic and static (turned into fake sequences) (10)
#lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
#model_pos, model_neg = lstmcl.train(trainB_dynamic_and_static_as_dynamic, trainB_labels)
#scores[10].append(lstmcl.test(model_pos, model_neg, test_dynamic_and_static_as_dynamic, test_labels))

# RF on static and dynamic (spatialized) features (11)
#rf = RandomForestClassifier(n_estimators=nestimators)
#rf.fit(trainB_static_and_dynamic_as_static, trainB_labels_all)
#scores[11].append(rf.score(test_static_and_dynamic_as_static, test_labels_all))

print "===> (2) Ensemble (RF) on predictions by RF and LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores[2]), np.std(scores[2]), scores[2])
print "===> (4) Hybrid (RF) on features enriched by LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores[4]), np.std(scores[4]), scores[4])
print "===> (8) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[8]), np.std(scores[8]), scores[8])
#print "===> (10) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[10]), np.std(scores[10]), scores[10])
#print "===> (11) RF on dynamic (spatialized) features: %.4f (+/- %.4f) %s" % (np.mean(scores[11]), np.std(scores[11]), scores[11])


