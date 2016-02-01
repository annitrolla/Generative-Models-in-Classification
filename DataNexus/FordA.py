import numpy as np

# load dataset
print 'Loading data...'
train_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/FordA/raw/FordA_TRAIN', delimiter=',')
test_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/FordA/raw/FordA_TEST', delimiter=',')
all_raw = np.concatenate((train_raw, test_raw))

# extract labels
print 'Transforming data...'
labels = all_raw[:, 0].astype(int);
labels[labels == -1] = 0
static = all_raw[:, 1:]
dynamic = static.reshape(all_raw.shape[0], 1, all_raw.shape[1] - 1)

# store data
print 'Storing data...'
np.save('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_static.npy', static)
np.save('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_dynamic.npy', dynamic)
np.save('/storage/hpc_anna/GMiC/Data/FordA/preprocessed/train_labels.npy', labels)

