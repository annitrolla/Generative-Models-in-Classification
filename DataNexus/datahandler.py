"""
Handles ECoG dataset
"""

import numpy as np

class DataHandler:

    @staticmethod
    def split(ratio, data, labels):
        idx_train = np.random.choice(range(0, data.shape[0]), size=int(data.shape[0] * ratio), replace=False)
        idx_val = list(set(range(0, data.shape[0])) - set(idx_train))
        train_data = data[idx_train, :, :]
        train_labels = labels[idx_train]
        val_data = data[idx_val, :, :]
        val_labels = labels[idx_val]    
        return train_data, train_labels, val_data, val_labels

