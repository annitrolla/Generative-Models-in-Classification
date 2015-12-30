# load MNIST dataset
from sklearn.datasets import fetch_mldata
import numpy as np

print 'Downloading MNIST...'
mnist = fetch_mldata("MNIST original")

digit_a = 3
digit_b = 5

# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# for binary classification we pick two digets
train_idx_a = np.where(y_train == digit_a)
train_idx_b = np.where(y_train == digit_b)
test_idx_a = np.where(y_test == digit_a)
test_idx_b = np.where(y_test == digit_b)

# static data
print 'Forming static data...'
static_train = np.concatenate((X_train[train_idx_a], X_train[train_idx_b]))
static_test = np.concatenate((X_test[test_idx_a], X_test[test_idx_b]))

# dynamic data
print 'Forming dynamic data'
cl = static_train.reshape(static_train.shape[0], 28, 28)
rw = np.transpose(cl, (0, 2, 1))
dynamic_train = np.concatenate((cl, rw), axis=1)

cl = static_test.reshape(static_test.shape[0], 28, 28)
rw = np.transpose(cl, (0, 2, 1))
dynamic_test = np.concatenate((cl, rw), axis=1)

# labels
print 'Forming labels...'
labels_train = np.concatenate((y_train[train_idx_a], y_train[train_idx_b])).astype(int)
labels_test = np.concatenate((y_test[test_idx_a], y_test[test_idx_b])).astype(int)

# assign 0 and 1 instead of original labels
labels_train[labels_train == digit_a] = 0
labels_train[labels_train == digit_b] = 1
labels_test[labels_test == digit_a] = 0
labels_test[labels_test == digit_b] = 1

# shuffle the data
nsamples_train = len(labels_train)
nsamples_test = len(labels_test)
new_train_idx = np.random.choice(range(0, nsamples_train), size=nsamples_train, replace=False)
new_test_idx = np.random.choice(range(0, nsamples_test), size=nsamples_test, replace=False)

static_train = static_train[new_train_idx]
dynamic_train = dynamic_train[new_train_idx]
labels_train = labels_train[new_train_idx]

static_test = static_test[new_test_idx]
dynamic_test = dynamic_test[new_test_idx]
labels_test = labels_test[new_test_idx]

# store the data
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_static.npy', static_train)
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_dynamic.npy', dynamic_train)
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_static.npy', static_test)
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_dynamic.npy', dynamic_test)
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/train_labels.npy', labels_train)
np.save('/storage/hpc_anna/GMiC/Data/mnist/preprocessed/test_labels.npy', labels_test)

