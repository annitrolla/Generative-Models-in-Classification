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
from DataNexus.datahandler import DataHandler

batch_size = 32

# read the data
print("Reading data...")
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
#test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
#test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

# put features to be the last dimension
X_train = np.transpose(train_data, (0, 2, 1))
#X_test = np.transpose(X_test, (0, 2, 1))

print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(64, 128))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, train_labels, batch_size=batch_size, nb_epoch=10, validation_split=0.3, show_accuracy=True)
#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
#print('Test score:', score)
#print('Test accuracy:', acc)
