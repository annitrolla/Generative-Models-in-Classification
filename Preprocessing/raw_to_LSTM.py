# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:03:45 2015

@author: annaleontjeva
"""
import numpy as np
#import matplotlib.pyplot as plt


data_train = np.loadtxt('../../Data/ECoG/Competition_train_cnt.txt')
data_test = np.loadtxt('../../Data/ECoG/Competition_test_cnt.txt')

data_labels = np.loadtxt('../../Data/ECoG/Competition_train_lab.txt')

#data_norm_train = (data_train-np.min(data_train))/np.absolute(np.max(data_train) - np.min(data_train))
#data_norm_test = (data_test-np.min(data_test))/np.absolute(np.max(data_test) - np.min(data_test))

data_norm_train = (data_train - np.mean(data_train.flatten()))/np.std(data_train.flatten())
data_norm_test = (data_test - np.mean(data_test.flatten()))/np.std(data_test.flatten())

data_round_train = (np.round(data_norm_train*100, 0) + 4000).astype(int)
data_round_test = (np.round(data_norm_test*100, 0) + 4000).astype(int)

reshaped_train = np.reshape(data_round_train, (278, 64, 3000))

# we take channel 0 only

# create validation
train_idx = np.random.choice(range(0, 278), size=np.round(278*0.70,0), replace=False)
val_idx = list(set(range(0, 278))- set(train_idx))
train = reshaped_train[train_idx, :, :]
val = reshaped_train[val_idx, :, :]

train_labels = data_labels[train_idx]
val_labels = data_labels[val_idx]

train_pos = train[train_labels == 1,:,:]
train_neg = train[train_labels == -1,:,:]

train_data_pos = np.reshape(train_pos, (train_pos.shape[0]*train_pos.shape[1], train_pos.shape[2]))
train_data_neg = np.reshape(train_neg, (train_neg.shape[0]*train_neg.shape[1], train_neg.shape[2]))

val_data = np.reshape(val, (val.shape[0]*val.shape[1], val.shape[2]))

np.savetxt('../../Data/ECoG/full/input.txt', data_round_train, fmt='%d', delimiter='', newline='||||')

np.savetxt('../../Data/ECoG/pos/input.txt', train_data_pos, fmt='%d', delimiter='', newline='||||')
np.savetxt('../../Data/ECoG/neg/input.txt', train_data_neg, fmt='%d', delimiter='', newline='||||')

np.savetxt('../../Data/ECoG/val/input.txt', val_data, fmt='%d', delimiter='', newline='||||')

#np.savetxt('../../Data/ECoG/lstmtrain_labels.txt', train_labels, fmt='%d', delimiter='', newline='\n')
np.savetxt('../../Data/ECoG/val/labels.txt', val_labels, fmt='%d', delimiter='', newline='\n')

np.savetxt('../../Data/ECoG/lstmtest.txt', data_round_test, fmt='%d', delimiter='', newline='||||')

#print np.min(data_round_train), np.max(data_round_train)
#print np.min(data_round_test), np.max(data_round_test)

#plt.hist(data_round_train.flatten(), bins=20)
#plt.show()

#plt.hist(data_round_test.flatten(), bins=20)
#plt.show()
#------------------------------------------------#
# we take channel 0 only

reshaped_train_chn_0 = reshaped_train[:,0,:]
# create validation
train_idx = np.random.choice(range(0, 278), size=np.round(278*0.70,0), replace=False)
val_idx = list(set(range(0, 278))- set(train_idx))
train_chn_0 = reshaped_train_chn_0[train_idx, :]
val_chn_0 = reshaped_train_chn_0[val_idx, :]

train_labels_chn_0 = data_labels[train_idx]
val_labels_chn_0 = data_labels[val_idx]

train_pos_chn_0 = train_chn_0[train_labels == 1,:]
train_neg_chn_0 = train_chn_0[train_labels == -1,:]

np.savetxt('../../Data/ECoG/full/input.txt', reshaped_train_chn_0, fmt='%d', delimiter='', newline='||||')
np.savetxt('../../Data/ECoG/pos/input.txt', train_pos_chn_0, fmt='%d', delimiter='', newline='||||')
np.savetxt('../../Data/ECoG/neg/input.txt', train_neg_chn_0, fmt='%d', delimiter='', newline='||||')
np.savetxt('../../Data/ECoG/val/input.txt', val_chn_0, fmt='%d', delimiter='', newline='||||\n')
np.savetxt('../../Data/ECoG/val/labels.txt', val_labels_chn_0, fmt='%d', delimiter='', newline='\n')
