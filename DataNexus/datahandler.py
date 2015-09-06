"""
Handles ECoG dataset
"""

import numpy as np

class DataHandler:

    trainval_data = None
    trainval_labels = None
    test_data = None
    test_labels = None
    datapath = None

    def __init__(self, datapath):
        self.datapath = datapath
                
    def load_train_data(self):
        print "Loading trainval data..."
        self.trainval_data = np.load(self.datapath + "/train_data.npy")
        self.trainval_labels = np.load(self.datapath + "/train_labels.npy")
 
    def load_test_data(self):
        print "Loading test data..."
        self.test_data = np.load(self.datapath + "/test_data.npy")
        self.test_labels = np.load(self.datapath + "/test_labels.npy")

    def split_train(self, ratio):
        idx_train = np.random.choice(range(0, self.trainval_data.shape[0]), size=int(self.trainval_data.shape[0] * ratio), replace=False)
        idx_val = list(set(range(0, self.trainval_data.shape[0])) - set(idx_train))
        train_data = self.trainval_data[idx_train, :, :]
        train_labels = self.trainval_labels[idx_train]
        val_data = self.trainval_data[idx_val, :, :]
        val_labels = self.trainval_labels[idx_val]    
        return train_data, train_labels, val_data, val_labels

