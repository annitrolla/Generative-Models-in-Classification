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
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from DataNexus.datahandler import DataHandler
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2

class LSTMClassifier:
    
    def __init__(self, lstmsize, dropout, optim, nepoch, batch_size, validation_split=None):
        self.lstmsize = lstmsize
        self.dropout = dropout
        self.optim = optim
        self.nepoch = nepoch
        self.batch_size = batch_size
        self.validation_split = validation_split

    @staticmethod
    def sequence_lag(data):
        X = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        y = np.empty((data.shape[0], data.shape[1]-1, data.shape[2]), dtype=np.float32)
        for i in range(data.shape[0]):
            X[i, :, :] = data[i, :-1, :] # a shorter sequence is inserted at the end of the fixed length matrix
            y[i, :, :] = data[i, 1:, :]
        return X, y

    def build_model(self, data):
        # prepare training and test set as follows:
        #   sequences in the training set will miss one observation in the end [1...299]
        #             (299th observation will be used to predict the 300th)
        #   sequences in the test set will miss one observation from the begining [2...300]
        #             (there is nothing to predict the first observation from)
        data = np.transpose(data, (0, 2, 1))
        #X_train, y_train = self.sequence_lag(data)
        X_train = data[:, :-1, :]
        y_train = data[:, 1:, :]

	seqlen = data.shape[1]
	nfeatures = data.shape[2]

        print('Build model...')
        model = Sequential()
        model.add(LSTM(self.lstmsize, return_sequences=True, input_shape=(seqlen, nfeatures)))
        model.add(Dropout(self.dropout))
        model.add(TimeDistributedDense(y_train.shape[2]))
        model.add(Activation('relu'))

        print('Compiling model...')
        model.compile(loss='mean_squared_error', optimizer=self.optim)

        print("Training...")
        model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=self.nepoch, validation_split=self.validation_split)
        
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

    def activations(self, model, data):
        """
        The idea is to build a model which is identical to the actual, but does not have the last layer
        https://github.com/fchollet/keras/issues/41
        """
        data = np.transpose(data, (0, 2, 1))
	seqlen = data.shape[1]
        nfeatures = data.shape[2]

        extractor = Sequential()
        extractor.add(LSTM(self.lstmsize, return_sequences=True, input_shape=(seqlen, nfeatures), weights=model.layers[0].get_weights()))

        extractor.compile(loss='categorical_crossentropy', optimizer=self.optim, class_mode='categorical')
        activations = extractor.predict(data, batch_size=self.batch_size)

        return activations

class LSTMDiscriminative:
    
    def __init__(self, lstmsize, fcsize, dropout, optim, nepoch, batch_size, validation_split=None, validation_data=None):
        self.lstmsize = lstmsize
        self.fcsize = fcsize
        self.dropout = dropout
        self.optim = optim
        self.nepoch = nepoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.validation_data = validation_data

    def train(self, data, labels):
        print('Training LSTMDiscriminative model')
        data = np.transpose(data, (0, 2, 1))

        # encode labels into one-hot
        enc = OneHotEncoder(sparse=False)
        labels = enc.fit_transform(np.matrix(labels).T)

        print('    Building the model...')
        model = Sequential()
        model.add(LSTM(data.shape[2], self.lstmsize, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.lstmsize, self.fcsize, activation='relu', W_regularizer=l2(0.01)))
        #model.add(Dropout(self.dropout))
        model.add(Dense(self.fcsize, 2, activation='softmax', W_regularizer=l2(0.01)))

        print('    Compiling the model...')
        model.compile(loss='categorical_crossentropy', optimizer=self.optim, class_mode='categorical')

        print("    Training the model...")
        model.fit(data, labels, batch_size=self.batch_size, nb_epoch=self.nepoch, show_accuracy=True, validation_split=self.validation_split, validation_data=self.validation_data)

        return model 

    def test(self, model, data, labels):
        data = np.transpose(data, (0, 2, 1))
        enc = OneHotEncoder(sparse=False)
        labels = enc.fit_transform(np.matrix(labels).T)
        loss, accuracy = model.evaluate(data, labels, batch_size=self.batch_size, show_accuracy=True)
        return accuracy

    def pos_neg_ratios(self, model, data):
        data = np.transpose(data, (0, 2, 1))
        predicted = model.predict(data)
        ratios = np.log(predicted[:, 1] / predicted[:, 0])
        return ratios

    def activations(self, model, data):
        """
        The idea is to build a model which is identical to the actual, but does not have the last layer
        https://github.com/fchollet/keras/issues/41
        """
        data = np.transpose(data, (0, 2, 1))

        extractor = Sequential()
        extractor.add(LSTM(data.shape[2], self.lstmsize, return_sequences=False,
                           weights=model.layers[0].get_weights()))
        extractor.add(Dense(self.lstmsize, self.fcsize, activation='relu',
                            weights=model.layers[2].get_weights()))
        extractor.compile(loss='categorical_crossentropy', optimizer=self.optim, class_mode='categorical')
        activations = extractor.predict(data, batch_size=self.batch_size)
        return activations

