import numpy as np

# load dataset
print 'Loading data...'
train_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/Yoga/raw/Yoga_TRAIN', delimiter=',')
test_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/Yoga/raw/Yoga_TEST', delimiter=',')
all_raw = np.concatenate((train_raw, test_raw))

# extract labels
print 'Transforming data...'

# train
train_static = train_raw[:, 1:]
train_dynamic = train_static.reshape(train_raw.shape[0], 1, train_raw.shape[1] - 1)
train_labels = train_raw[:, 0].astype(int);
train_labels[train_labels == 1] = 0
train_labels[train_labels == 2] = 1

# test
test_static = test_raw[:, 1:]
test_dynamic = test_static.reshape(test_raw.shape[0], 1, test_raw.shape[1] - 1)
test_labels = test_raw[:, 0].astype(int);
test_labels[test_labels == 1] = 0
test_labels[test_labels == 2] = 1

# together for CV
all_static = all_raw[:, 1:]
all_dynamic = all_static.reshape(all_raw.shape[0], 1, all_raw.shape[1] - 1)
all_labels = all_raw[:, 0].astype(int);
all_labels[all_labels == 1] = 0
all_labels[all_labels == 2] = 0

# store data
print 'Storing data...'
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/train_labels.npy', train_labels)

np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/test_labels.npy', test_labels)

np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/all_static.npy', all_static)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/all_dynamic.npy', all_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Yoga/preprocessed/all_labels.npy', all_labels)

