"""
Handles ECoG dataset
"""

import numpy as np 


class ECoG:

    trainval_data = None
    trainval_labels = None
    test_data = None
    test_labels = None

    def load_train_data(self):
        raw_data = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_cnt_scaled.txt")
        self.trainval_data = np.reshape(raw_data, (278, 64, 3000))
        self.trainval_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_lab_onezero.txt")

    def load_test_data(self):
        raw_data = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_cnt_scaled.txt")
        self.test_data = np.reshape(raw_data, (100, 64, 3000))
        self.test_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_lab_onezero.txt")

    def split_train_val(self, ratio=0.7):
        idx_train = np.random.choice(range(0, data.shape[0]), size=int(data.shape[0]*ratio), replace=False)
        idx_val = list(set(range(0, data.shape[0]))- set(idx_train))
        train_data = data[idx_train, :, :]
        train_labels = labels[idx_train]
        val_data = data[idx_val, :, :]
        val_labels = labels[idx_val]    
        return train_data, train_labels, val_data, val_labels

