"""

Preprocess Kaggle Winton Stock Market Challenge dataset
(https://www.kaggle.com/c/the-winton-stock-market-challenge)
into statical and dynamical features and split training and test

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we use only training data from the competition as we need
# to have true answers to evaluate the performance
print 'Reading the data...'
data = pd.read_csv('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/raw/train.csv')
nsamples = data.shape[0]
nfeatures = 26
nseqfeatures = 1
seqlen = 178

# first 25 features are static, reaplce missing values with 0
print 'Extracting static features...'
static_all = data.iloc[:, 1:27]
static_all = np.array(static_all.fillna(0))

# next 178 features are time-series of stock returns per minute
print 'Extracting dynamic features...'
dynamic_all = data.iloc[:, 28:206].fillna(0)
dynamic_all = np.array(dynamic_all).reshape(nsamples, nseqfeatures, seqlen)

# as label we take whether the return on the next day was positive or negative
print 'Extracting labels...'
labels_all = 1 * (data.iloc[:, 207] > 0.0)

# split training and test
print 'Splitting training and test sets...'
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))
train_static = static_all[train_idx, :]
train_dynamic = dynamic_all[train_idx, :, :]
test_static = static_all[test_idx, :]
test_dynamic = dynamic_all[test_idx, :, :]
train_labels = labels_all[train_idx]
test_labels = labels_all[test_idx]

# store the dataset
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/train_labels.npy', train_labels)
np.save('/storage/hpc_anna/GMiC/Data/Kaggle-Winton-Stock/preprocessed/test_labels.npy', test_labels)

