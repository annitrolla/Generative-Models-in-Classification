# -*- coding: utf-8 -*-
"""
Normalizing data within each subject

Created on Sun Jul 26 15:13:12 2015

@author: annaleontjeva
"""
import numpy as np

# load the data
raw_data = np.loadtxt('../../Data/ECoG/Competition_train_cnt.txt')
raw_labels = np.loadtxt('../../Data/ECoG/Competition_train_lab.txt')

# map -1 and 1 to 0-1 and add to the train data
trainval_labels = (raw_labels + 1)/2

for r in range(raw_data.shape[0]): 
    raw_data[r, :] = (raw_data[r, :] - np.mean(raw_data[r, :]))/np.std(raw_data[r, :])


np.savetxt('../../Data/ECoG/Competition_train_cnt_scaled.txt', raw_data, fmt='%f')
np.savetxt('../../Data/ECoG/Competition_train_lab_onezero.txt', trainval_labels, fmt='%d')