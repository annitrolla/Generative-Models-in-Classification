"""

Classifier based on generative LSTM applied to syn_lstm_wins

"""

import numpy as np
from keras.optimizers import RMSprop
from LSTM.lstm_classifier import LSTMClassifier

# parameters
lstmsize = 2000
lstmdropout = 0.0
#lstmoptim = 'rmsprop' # Default RMSProp is lr=0.001, rho=0.9, epsilon=1e-6
#lstmoptim = RMSprop(lr=0.01, rho=0.9, epsilon=1e-6)
lstmoptim = 'adadelta'
lstmnepochs = 100
lstmbatch = 64

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split the data into training and test
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

# train the model and report performance
print 'Training the model...'
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatch, validation_split=0.3)
model_pos, model_neg = lstmcl.train(dynamic_train[train_idx], labels_train[train_idx])
print 'Generative LSTM classifier on dynamic features: %.4f' % lstmcl.test(model_pos, model_neg, dynamic_train[test_idx], labels_train[test_idx])

