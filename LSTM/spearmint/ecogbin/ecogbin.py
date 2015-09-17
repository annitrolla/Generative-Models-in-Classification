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
from LSTM.lstm_classifier import LSTMClassifier
import numpy as np
                                   
def ecoglstm(lstmsize, dropout, optim):
    batch_size = 16

    print("Reading data...")
    data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
    labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
    train_data, train_labels, val_data, val_labels = DataHandler.split(0.7, data, labels)
    
    lstmcl = LSTMClassifier(lstmsize[0], dropout[0], optim[0], 10)
    model_pos, model_neg = lstmcl.train(train_data, train_labels)
    result = -lstmcl.test(model_pos, model_neg, val_data, val_labels)
    
    print('Result = %f' % result)
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return ecoglstm(params['lstmsize'], params['dropout'], params['optim'])
