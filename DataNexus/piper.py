"""

Preprocess Plumbr dataset
into statical and dynamical features and split training and test

"""

import numpy as np
import matplotlib.pyplot as plt

# we use only training data from the competition  we need
# to have true answers to evaluate the performance print 'Reading the data...'
static = np.loadtxt('/storage/hpc_anna/GMiC/Data/Piper/raw/static_piper.txt', delimiter=',')
dynamic = np.loadtxt('/storage/hpc_anna/GMiC/Data/Piper/raw/dynamic_piper.txt', delimiter=',')
labels = np.loadtxt('/storage/hpc_anna/GMiC/Data/Piper/raw/labels_piper.txt', delimiter=',')

nsamples = static.shape[0]
nfeatures = 6
nseqfeatures = 17
seqlen = 10

dynamic_reshaped = dynamic.reshape(dynamic.shape[0],nseqfeatures,seqlen)
dynamic_reshaped[:,[15,16],:]=np.abs(dynamic_reshaped[:,[15,16],:])
#dynamic_reshaped[:,[3,4,5,6,7,8,9,10,13,14,15,16],:] = dynamic_reshaped[:,[3,4,5,6,7,8,9,10,13,14,15,16],:]/10000000
dynamic_reshaped = np.log(dynamic_reshaped)
dynamic_reshaped = np.nan_to_num(dynamic_reshaped)

# split training and test
print 'Splitting training and test sets...'
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))
train_static = static[train_idx, :]
train_dynamic = dynamic_reshaped[train_idx, :, :]
test_static = static[test_idx, :]
test_dynamic = dynamic_reshaped[test_idx, :, :]
train_labels = labels[train_idx]
test_labels = labels[test_idx]

# store the dataset
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_labels.npy', train_labels)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_labels.npy', test_labels)

