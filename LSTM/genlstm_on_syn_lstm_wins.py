import numpy as np
from LSTM.lstm_classifier import LSTMClassifier

# parameters
lstmsize = 512
lstmdropout = 0.0
lstmoptim = 'rmsprop'
lstmnepochs = 50
lstmbatch = 32

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split the data into training and test
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

# train the model and report performance
print 'Training the model...'
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatch)
model_pos, model_neg = lstmcl.train(dynamic_train[train_idx], labels_train[train_idx])
print 'Generative LSTM classifier on dynamic features: %.4f' % lstmcl.test(model_pos, model_neg, dynamic_train[test_idx], labels_train[test_idx])

