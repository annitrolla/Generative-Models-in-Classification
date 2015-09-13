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
        seq_range = range(1, data.shape[1], 100)
        X_train = np.empty((data.shape[0] * len(seq_range), data.shape[1]-1, data.shape[2]), dtype=np.float32)
        y_train = np.empty((data.shape[0] * len(seq_range), data.shape[2]), dtype=np.float32)
        
        counter = 0
        for s in range(data.shape[0]):
            for i in seq_range:    
                 X_train[counter, -i:, :] = data[s, 0:i, :] # a shorter sequence is inserted at the end of the fixed length matrix
                 y_train[counter, :] = data[s, i, :]
                 counter += 1
            print(counter)
        print(X_train.shape, y_train.shape)
        #X_train = np.transpose(data, (0, 2, 1))
        #y_train = np.transpose(data, (0, 2, 1))

        # TODO:
        # In the example they cut initial sequence is cut into 20-obs-len overlapping sequences
        # each sequence is trained to predict the next (21st) observation. In our case it is not
        # clear whether this way is applicable: do we learn the whole temporal dynamics of the
        # sequence if we do that trick?
        # An alternative: create 299 sequences of increasing lengths (1, 2, 3, ..., 299) and train
        # each to predict the next character (2nd, 3rd, 4th, ..., 300th)
        
        print('Build model...')
        model = Sequential()
        model.add(LSTM(data.shape[2], 256, return_sequences=False))  # Spearmint parameter: size
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
    

