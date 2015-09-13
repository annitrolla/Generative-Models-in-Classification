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
from DataNexus.datahandler import DataHandler

class LSTMClassifier:
    
    def __init__(self, lstmsize, dropout, optim, nepoch):
        self.lstmsize = lstmsize
        self.dropout = dropout
        self.optim = optim
        self.nepoch = nepoch
    @staticmethod
    def sequence_lag(data):
        X = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        y = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        for i in range(data.shape[0]):
            X[i, :, :] = data[i, :-1, :] # a shorter sequence is inserted at the end of the fixed length matrix
            y[i, :, :] = data[i, 1:, :]
        return X, y

    def build_model(self, data):

        batch_size = 256

        # prepare training and test set as follows:
        #   sequences in the training set will miss one observation in the end [1...299]
        #             (299th observation will be used to predict the 300th)
        #   sequences in the test set will miss one observation from the begining [2...300]
        #             (there is nothing to predict the first observation from)
        data = np.transpose(data, (0, 2, 1))
        X_train, y_train = self.sequence_lag(data)
        
        print('Build model...')
        model = Sequential()
        model.add(LSTM(data.shape[2], self.lstmsize, return_sequences=True))  # Spearmint parameter: size
        model.add(Dropout(self.dropout))              # Spearmint parameter: dropout
        model.add(Dense(self.lstmsize, data.shape[2]))
        model.add(Activation('relu'))        # Spearmint parameter: activation

        print('Compiling model...')
        model.compile(loss='mean_squared_error', optimizer=self.optim) # Spearmint parameter: optimizer

        print("Training...")
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=self.nepoch, validation_split=0.3, show_accuracy=True)       
        
        return model
    
    def train(self, data, labels):
        train_pos = data[labels==1, :, :]
        train_neg = data[labels==0, :, :]
       
        print("Start training positive model...")
        model_pos = self.build_model(train_pos)
       
        print("Start training positive model...")
        model_neg = self.build_model(train_neg)
        
        return model_pos, model_neg

    def pos_neg_ratios(self, model_pos, model_neg, data):
        data = np.transpose(data, (0, 2, 1))
        X, y = self.sequence_lag(data)
        predicted_pos = model_pos.predict(X)
        predicted_neg = model_neg.predict(X)
        mse_pos = np.sum((predicted_pos - y)**2, axis=(1,2)) / (y.shape[1] * y.shape[2])
        mse_neg = np.sum((predicted_neg - y)**2, axis=(1,2)) / (y.shape[1] * y.shape[2])
        
        # ratio shows how much the positive model is better than the negative one, thus, we take inverse of mse
        # log transforms the measure to be symmetrical around zero
        ratios = np.log(mse_neg / mse_pos) 
        
        return ratios

    def test(self, model_pos, model_neg, data, labels):
        ratios = self.pos_neg_ratios(model_pos, model_neg, data)
        predictions = (ratios >= 0) * 1
        accuracy = np.sum(predictions == labels) / float(len(labels))
        return accuracy

if __name__ == '__main__':

    print("Reading data...")
    train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
    train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
    test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
    test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")
    
    lstmcl = LSTMClassifier(256, 0.5, 'rmsprop', 20)
    model_pos, model_neg = lstmcl.train(train_data, train_labels)
    print(lstmcl.test(model_pos, model_neg, test_data, test_labels))



