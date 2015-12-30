"""

Train the whole set of test on the "lstm wins" synthetic dataset
 

"""

import numpy as np
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier 

# general parameters
nfolds = 5
nestimators = 500
lstmsize = 2000
lstmdropout = 0.0
lstmoptim = 'adadelta'
lstmnepochs = 50
lstmbatchsize = 64

#
# Load the dataset
#
print 'Loading the dataset..'
static_all = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_static.npy')
dynamic_all = np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_dynamic.npy')
labels_all = np.ravel(np.load('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_labels.npy'))
nsamples = static_all.shape[0]


#
# k-fold CV
#

# prepare where to store the predictions
predictions_all_rf = np.empty((len(labels_all), 2))
predictions_all_lstm = np.empty((len(labels_all), 2))

# prepare where to store the ratios
ratios_all_lstm = np.empty(len(labels_all))

# split indices into folds
predict_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
for fid, predict_idx in enumerate(predict_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(predict_idx))
    
    # extract predictions using RF on static
    print "    Extracting predictions on static data with RF..."
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(static_all[train_idx], labels_all[train_idx])
    predictions_all_rf[predict_idx] = rf.predict_log_proba(static_all[predict_idx])
    predictions_all_rf[predictions_all_rf == -inf] = np.min(predictions_all_rf[predictions_all_rf != -inf])

    # extract predictions using LSTM on dynamic
    print "    Extracting predictions on dynamic data with LSTM..."
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
    mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, dynamic_all[predict_idx]) 
    predictions_all_lstm[predict_idx] = np.vstack((mse_pos, mse_neg)).T
    ratios_all_lstm[predict_idx] = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[predict_idx])


#
# Prepare combined datasets for the future experiments
#

# datasets for ensemble learning
predictions_combined_rf_lstm = np.concatenate((predictions_all_rf, predictions_all_lstm), axis=1)

# datasets for hybrid learning
enriched_by_lstm = np.concatenate((static_all, np.matrix(ratios_all_lstm).T), axis=1)

# dataset to check how generative models perform if provided with static features along with dynamic
static_as_dynamic = np.zeros((static_all.shape[0], static_all.shape[1], dynamic_all.shape[2]))
for i in range(static_all.shape[0]):
    static_as_dynamic[i, :, :] = np.tile(static_all[i, :], (dynamic_all.shape[2], 1)).T
dynamic_and_static_as_dynamic = np.concatenate((dynamic_all, static_as_dynamic + np.random.uniform(-0.0001, 0.0001, static_as_dynamic.shape)), axis=1)


#
# k-fold CV for performance estimation
#
val_idx_list = np.array_split(range(nsamples), nfolds)
scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d / %d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # Ensemble on predictions by RF and LSTM (2)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(predictions_combined_rf_lstm[train_idx], labels_all[train_idx])
    scores[2].append(rf.score(predictions_combined_rf_lstm[val_idx], labels_all[val_idx]))

    # Hybrid on features enriched by LSTM (4)
    rf = RandomForestClassifier(n_estimators=nestimators)
    rf.fit(enriched_by_lstm[train_idx], labels_all[train_idx])
    scores[4].append(rf.score(enriched_by_lstm[val_idx], labels_all[val_idx]))

    # LSTM on dynamic features (8)
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
    scores[8].append(lstmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx]))

    # LSTM on dynamic and static (turned into fake sequences) (10)
    lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
    model_pos, model_neg = lstmcl.train(dynamic_and_static_as_dynamic[train_idx], labels_all[train_idx])
    scores[10].append(lstmcl.test(model_pos, model_neg, dynamic_and_static_as_dynamic[val_idx], labels_all[val_idx]))

print "===> (2) Ensemble (RF) on predictions by RF and LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores[2]), np.std(scores[2]), scores[2])
print "===> (4) Hybrid (RF) on features enriched by LSTM: %.4f (+/- %.4f) %s" % (np.mean(scores[4]), np.std(scores[4]), scores[4])
print "===> (8) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[8]), np.std(scores[8]), scores[8])
print "===> (10) LSTM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores[10]), np.std(scores[10]), scores[10])


