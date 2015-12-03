"""

Classifier based on generative LSTM applied to syn_lstm_wins

"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adadelta
from LSTM.lstm_classifier import LSTMDiscriminative

# parameters
lstmsize = 128
fcsize = 100
dropout = 0.5
optim = Adadelta(lr=0.1, rho=0.95, epsilon=1e-6) #'adadelta'
nepochs = 100
batchsize = 128

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split the data into training and test
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
val_idx = list(set(range(0, nsamples)) - set(train_idx))
dynamic_val = dynamic_train[val_idx]
labels_val = labels_train[val_idx]
dynamic_train = dynamic_train[train_idx]
labels_train = labels_train[train_idx]

# train the model and report performance
lstmcl = LSTMDiscriminative(lstmsize, fcsize, dropout, optim, nepochs, batchsize, validation_split=0.3)
model = lstmcl.train(dynamic_train, labels_train)
print 'Generative LSTM classifier on dynamic features: %.4f' % lstmcl.test(model, dynamic_val, labels_val)

#dynamic_val = np.transpose(dynamic_val, (0, 2, 1))
#enc = OneHotEncoder(sparse=False)
#labels_val = enc.fit_transform(np.matrix(labels_val).T)

#lstmcl = LSTMDiscriminative(lstmsize, fcsize, dropout, optim, nepochs, batchsize, validation_data=(dynamic_val, labels_val))
#model = lstmcl.train(dynamic_train, labels_train)
#print 'Generative LSTM classifier on dynamic features: %.4f' % lstmcl.test(model, dynamic_val, labels_val)
