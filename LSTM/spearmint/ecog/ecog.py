from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from DataNexus.datahandler import DataHandler
import numpy as np
                                   
def ecoglstm(lstmsize, dropout, optim):
    batch_size = 32

    print("Reading data...")
    train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
    train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
    #test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
    #test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

    # put features to be the last dimension
    X_train = np.transpose(train_data, (0, 2, 1))
    
    print("Building model...")
    model = Sequential()
    model.add(LSTM(64, lstmsize[0]))
    model.add(Dropout(dropout[0]))
    model.add(Dense(lstmsize[0], 1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    print("Training...")
    model.compile(loss='binary_crossentropy', optimizer=optim[0], class_mode="binary")
    results = model.fit(X_train, train_labels, batch_size=batch_size, nb_epoch=10, validation_split=0.3, show_accuracy=True)
    result = results.history['val_acc'][-1]
    print('Result = %f' % result)
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return ecoglstm(params['lstmsize'], params['dropout'], params['optim'])
