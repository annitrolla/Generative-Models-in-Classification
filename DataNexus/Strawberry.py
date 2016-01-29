import numpy as np

# load dataset
print 'Loading data...'
train_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/Strawberry/raw/Strawberry_TRAIN', delimiter=',')
test_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/Strawberry/raw/Strawberry_TEST', delimiter=',')
all_raw = np.concatenate((train_raw, test_raw))

# extract labels
print 'Transforming data...'
labels = all_raw[:, 0].astype(int);
labels[labels == 2] = 0
dynamic = all_raw[:, 1:].reshape(all_raw.shape[0], 1, all_raw.shape[1] - 1)

# store data
print 'Storing data...'
np.save('/storage/hpc_anna/GMiC/Data/Strawberry/preprocessed/train_dynamic.npy', dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Strawberry/preprocessed/train_labels.npy', labels)

