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
from sklearn.preprocessing import OneHotEncoder

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

        batch_size = 64

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
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=self.nepoch, show_accuracy=True)       
        
        return model
    
    def train(self, data, labels):
        train_pos = data[labels==1, :, :]
        train_neg = data[labels==0, :, :]
       
        print("Start training positive model...")
        model_pos = self.build_model(train_pos)
       
        print("Start training positive model...")
        model_neg = self.build_model(train_neg)
        
        return model_pos, model_neg

    def predict_mse(self, model_pos, model_neg, data):
        data = np.transpose(data, (0, 2, 1))
        X, y = self.sequence_lag(data)
        predicted_pos = model_pos.predict(X)
        predicted_neg = model_neg.predict(X)
        mse_pos = np.sum((predicted_pos - y)**2, axis=(1,2)) / (y.shape[1] * y.shape[2])
        mse_neg = np.sum((predicted_neg - y)**2, axis=(1,2)) / (y.shape[1] * y.shape[2])
        return mse_pos, mse_neg

    def pos_neg_ratios(self, model_pos, model_neg, data):
        # ratio shows how much the positive model is better than the negative one, thus, we take inverse of mse
        # log transforms the measure to be symmetrical around zero
        mse_pos, mse_neg = self.predict_mse(model_pos, model_neg, data)
        ratios = np.log(mse_neg / mse_pos) 
        return ratios

    def test(self, model_pos, model_neg, data, labels):
        ratios = self.pos_neg_ratios(model_pos, model_neg, data)
        predictions = (ratios >= 0) * 1
        accuracy = np.sum(predictions == labels) / float(len(labels))
        return accuracy


class LSTMDiscriminative:
    
    def __init__(self, lstmsize, dropout, optim, nepoch, batch_size):
        self.lstmsize = lstmsize
        self.dropout = dropout
        self.optim = optim
        self.nepoch = nepoch
        self.batch_size = batch_size

    def train(self, data, labels):
        print('Training LSTMDescriminative model')
        data = np.transpose(data, (0, 2, 1))

        print('    Building the model...')
        model = Sequential()
        model.add(LSTM(data.shape[2], self.lstmsize, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.lstmsize, 1))
        model.add(Activation('sigmoid'))

        print('    Compiling the model...')
        model.compile(loss='binary_crossentropy', optimizer=self.optim)

        print("    Training the model...")
        model.fit(data, labels, batch_size=self.batch_size, nb_epoch=self.nepoch, show_accuracy=True)

        return model 

    def test(self, model, data, labels):
        data = np.transpose(data, (0, 2, 1))
        loss, accuracy = model.evaluate(data, labels, batch_size=self.batch_size, validation_split=0.3, show_accuracy=True)
        return accuracy


if __name__ == '__main__':

    # load the data
    static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
    dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
    static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
    dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
    labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
    labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')

    # train the model
    lstmcl = LSTMDiscriminative(500, 0.5, 'rmsprop', nepoch=5, batch_size=384)
    model = lstmcl.train(dynamic_train, labels_train)
    print(lstmcl.test(model, dynamic_val, labels_val))



