from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

batch_size = 32

X_train = np.random.randn(2000, 100, 1)
y_train = np.random.random(2000) > 0.5
X_test = np.random.randn(500, 100, 1)
y_test = np.random.random(500) > 0.5

# read the data
print("Reading data...")
X_train = np.loadtxt('/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_cnt_scaled.txt')
y_train = np.loadtxt('/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_lab_onezero.txt')
#X_test = np.loadtxt('/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_cnt_scaled.txt')
#y_test = np.loadtxt('/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_lab_onezero.txt')

# reshape
X_train = np.reshape(X_train, (278, 64, 3000))
#X_test = np.reshape(X_test, (100, 64, 3000))

# put features to be the last dimension
X_train = np.transpose(X_train, (0, 2, 1))
#X_test = np.transpose(X_test, (0, 2, 1))

# split training and validation
shuffle = np.random.permutation(X_train.shape[0])
ntrain = int(X_train.shape[0] * 0.7)
X_val = X_train[shuffle[(ntrain + 1):]]
y_val = y_train[shuffle[(ntrain + 1):]]
X_train = X_train[shuffle[:ntrain]]
y_train = y_train[shuffle[:ntrain]]

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
#print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(64, 128, truncate_gradient=1))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_val, y_val), show_accuracy=True)
#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
#print('Test score:', score)
#print('Test accuracy:', acc)
