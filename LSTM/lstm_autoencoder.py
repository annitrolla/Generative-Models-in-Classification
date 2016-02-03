import numpy as np
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder, TimeDistributedDense, Activation, RepeatVector
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt

# load the dataset
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')
nsamples = dynamic_train.shape[0]
nfeatures = dynamic_train.shape[1]
seqlen = dynamic_train.shape[2]

dynamic_train = np.transpose(dynamic_train, (0, 2, 1))

# parameters
lstmsize = 1000
lstmdropout = 0.5
lstmoptim = 'rmsprop'
lstmnepochs = 50
lstmbatch = 768

# Initial LSTM
#model = Sequential()
#model.add(LSTM(101, return_sequences=True, input_shape=(seqlen, nfeatures)))
#model.add(TimeDistributedDense(1001))
#model.add(LSTM(50, return_sequences=True))
#model.compile(loss='mse', optimizer='rmsprop')
#model.fit(dynamic_train, dynamic_train, batch_size=lstmbatch, nb_epoch=lstmnepochs, validation_split=0.3)
#reconstruction = model.predict(dynamic_train, batch_size=lstmbatch)


# Symmetric LSTM
#model = Sequential()
#model.add(LSTM(256, return_sequences=True, input_shape=(seqlen, nfeatures)))
#model.add(TimeDistributedDense(301))
#model.add(LSTM(256, return_sequences=True, input_shape=(seqlen, 301)))
#model.add(TimeDistributedDense(50))
#model.compile(loss='mse', optimizer='rmsprop')
#model.fit(dynamic_train, dynamic_train, batch_size=lstmbatch, nb_epoch=lstmnepochs, validation_split=0.3)
#reconstruction = model.predict(dynamic_train, batch_size=lstmbatch)

# LSTM with RepeatVector
model = Sequential()
model.add(LSTM(3500, return_sequences=False, input_shape=(seqlen, nfeatures))) #first screen: 256
#model.add(Dense(3500))
model.add(RepeatVector(seqlen))
model.add(LSTM(3500, return_sequences=True))
model.add(TimeDistributedDense(50))
model.compile(loss='mse', optimizer='rmsprop')
model.fit(dynamic_train, dynamic_train, batch_size=32, nb_epoch=50, validation_split=0.3)


# Plot of train and reconstruction matrix
plt.clf()
plt.subplot(2, 1, 1)
plt.plot(dynamic_train[56,0:4,:].T)
plt.subplot(2, 1, 2)
plt.plot(reconstruction[56,0:4,:].T)
plt.savefig('/home/hpc_anna1985/Research/Generative-Models-in-Classification/Results/images/sample_56_symmetric_autoencoder.png')


# split the data into training and test
#train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
#test_idx = list(set(range(0, nsamples)) - set(train_idx))

# train the model and report performance
#print 'Training the model...'
#lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatch)
#model_pos, model_neg = lstmcl.train(dynamic_train[train_idx], labels_train[train_idx])
#print 'Generative LSTM classifier on dynamic features: %.4f' % lstmcl.test(model_pos, model_neg, dynamic_train[test_idx], labels_train[test_idx])

