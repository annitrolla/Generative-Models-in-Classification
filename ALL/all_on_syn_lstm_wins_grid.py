"""

Explore all models' performance for the following parameters:
nsamples, nfeatures, nseqfeatures, seqlen

The list of models includes: 
RF on static, HMM on dynamic, RF on dynamic, LSTM on dynamic, 
RF+HMM on both, RF+LSTM on both, LSTM on both

"""

import numpy as np
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

# read parameters
nsamples = int(args.nsamples)
nfeatures = int(args.nfeatures)
nseqfeatures = int(args.nseqfeatures)
seqlen = int(args.seqlen)

# general parameters
lstm_nepochs = 20

# open file to store results
f = open('../../Results/grid_lstm_wins.csv', 'a')

# generate the dataset
static_train, dynamic_train, static_val, dynamic_val, labels_train, labels_val = generate_lstm_wins(nsamples, nfeatures, nseqfeatures, seqlen) 
    
# store dataset parameters
result = "%d, %d, %d, %d" % (nsamples, nfeatures, nseqfeatures, seqlen)

# static data with RF
rf = RandomForestClassifier(n_estimators=100)
rf.fit(static_train, labels_train)
rfstat = rf.score(static_val, labels_val)
print "Random Forest with static features on validation set: %.4f" % rfstat
result += ", %.4f" % rfstat
    
# dynamic data with HMM
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, dynamic_train, labels_train)
hmmdyn = hmmcl.test(model_pos, model_neg, dynamic_val, labels_val)   
print "HMM with dynamic features on validation set: %.4f" % hmmdyn
result += ", %.4f" % hmmdyn

# dynamic data with RF
print "Training RF on the dynamic dataset..."
dynamic_as_static_train = dynamic_train.reshape((dynamic_train.shape[0], dynamic_train.shape[1] * dynamic_train.shape[2]))
dynamic_as_static_val = dynamic_val.reshape((dynamic_val.shape[0], dynamic_val.shape[1] * dynamic_val.shape[2]))
rf = RandomForestClassifier(n_estimators=100)
rf.fit(dynamic_as_static_train, labels_train)
rfdyn = rf.score(dynamic_as_static_val, labels_val)
print "RF with dynamic features on validation set: %.4f" % rfdyn
result += ", %.4f" % rfdyn

# dynamic data with LSTM
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(dynamic_train, labels_train)
lstmdyn = lstmcl.test(model_pos, model_neg, dynamic_val, labels_val)
print "LSTM with dynamic features on validation set: %.4f" % lstmdyn
result += ", %.4f" % lstmdyn

#
# Evaluating Joint Models
#
print ""
print "Splitting data in two halves..."
fh_idx = np.random.choice(range(0, dynamic_train.shape[0]), size=np.round(dynamic_train.shape[0] * 0.5, 0), replace=False)
sh_idx = list(set(range(0, dynamic_train.shape[0])) - set(fh_idx))
fh_data = dynamic_train[fh_idx, :, :]
fh_labels = labels_train[fh_idx]
sh_data = dynamic_train[sh_idx, :, :]
sh_labels = labels_train[sh_idx]

# RF+HMM
print "Evaluating RF+HMM model:"

print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(3, 10, fh_data, fh_labels)

print "Extracting ratios based on the HMM model..."
sh_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val)

print "Merging static features and HMM-based ratios..."
enriched_sh_data = np.hstack((static_train[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=100)
rf.fit(enriched_sh_data, sh_labels)
rfhmmboth = rf.score(enriched_val_data, labels_val)
print "RF+HMM with enriched features on validation set: %.4f" % rfhmmboth
result += ", %.4f" % rfhmmboth
 
# RF+LSTM
print "Evaluating RF+LSTM model:"

print "Training LSTM classifier..."
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(fh_data, fh_labels)

print "Extracting ratios based on the LSTM model..."
sh_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
val_ratios = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_val)

print "Merging static features and LSTM-based ratios..."
enriched_sh_data = np.hstack((static_train[sh_idx, :], sh_ratios.reshape(len(sh_ratios), 1)))
enriched_val_data = np.hstack((static_val, val_ratios.reshape(len(val_ratios), 1)))

print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=100)
rf.fit(enriched_sh_data, sh_labels)
rflstmboth = rf.score(enriched_val_data, labels_val)
print "RF+LSTM with enriched features on validation set: %.4f" % rflstmboth
result += ", %.4f" % rflstmboth
    
# LSTM on both dynamic and static features
# transform static features into "fake" sequences
dynamized_static_train = np.zeros((static_train.shape[0], static_train.shape[1], dynamic_train.shape[2]))
for i in range(static_train.shape[0]):
    dynamized_static_train[i, :, :] = np.tile(static_train[i, :], (dynamic_train.shape[2], 1)).T
dynamized_static_val = np.zeros((static_val.shape[0], static_val.shape[1], dynamic_val.shape[2]))
for i in range(static_val.shape[0]):
    dynamized_static_val[i, :, :] = np.tile(static_val[i, :], (dynamic_val.shape[2], 1)).T

# meld dynamized static and dynamic features together
all_train = np.concatenate((dynamized_static_train, dynamic_train), axis=1)
all_val = np.concatenate((dynamized_static_val, dynamic_val), axis=1)

# dynamic data with LSTM
lstmcl = LSTMClassifier(2000, 0.5, 'adagrad', lstm_nepochs)
model_pos, model_neg = lstmcl.train(all_train, labels_train)
lstmboth = lstmcl.test(model_pos, model_neg, all_val, labels_val)
print "LSTM with dynamized static and dynamic features on validation set: %.4f" % lstmboth
result += ", %.4f" % lstmboth
    
print result
f.write(result + '\n')
f.close()
