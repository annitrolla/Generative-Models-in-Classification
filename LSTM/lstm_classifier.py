"""

Binary classifier based on LSTM

"""
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

class LSTMClassifier:
    
    @staticmethod
    def train(data):

        batch_size = 32

        # prepare training and test set as follows:
        #   sequences in the training set will miss one observation in the end [1...299]
        #             (299th observation will be used to predict the 300th)
        #   sequences in the test set will miss one observation from the begining [2...300]
        #             (there is nothing to predict the first observation from)
        data = np.transpose(data, (0, 2, 1))
        
        X_train = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        y_train = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        
        for i in range(data.shape[0]):    
            X_train[i, :, :] = data[i, :-1, :] # a shorter sequence is inserted at the end of the fixed length matrix
            y_train[i, :, :] = data[i, 1:, :]

        print(X_train.shape, y_train.shape)
        
        print('Build model...')
        model = Sequential()
        model.add(LSTM(data.shape[2], 256, return_sequences=True))  # Spearmint parameter: size
        model.add(Dropout(0.5))              # Spearmint parameter: dropout
        model.add(Dense(256, data.shape[2]))
        model.add(Activation('relu'))        # Spearmint parameter: activation

        print('Compiling model...')
        model.compile(loss='mean_squared_error', optimizer='adam') # Spearmint parameter: optimizer

        print("Training...")
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_split=0.3, show_accuracy=True)       
        
        return model


if __name__ == '__main__':

    print("Reading data...")
    train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
    # train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")

    LSTMClassifier.train(train_data)
    

