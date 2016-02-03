import numpy as np
from sklearn.preprocessing import OneHotEncoder

# load the data
print 'Loading data...'
train = np.loadtxt('/storage/hpc_anna/GMiC/Data/SpliceData/raw/splice_train.csv', delimiter=',')
labels = np.loadtxt('/storage/hpc_anna/GMiC/Data/SpliceData/raw/splice_labels.csv', delimiter=',')
nsamples = train.shape[0]

# convert labels to in
labels = labels.astype(int)

# one-hot encoding
enc = OneHotEncoder(sparse=False, n_values=8)
train_encoded = enc.fit_transform(train)
dynamic = np.transpose(train_encoded.reshape((1535,60,8)), axes=(0,2,1))
static = dynamic.reshape((dynamic.shape[0], dynamic.shape[1] * dynamic.shape[2]))

# shuffle
print 'Shuffling...'
new_train_idx = np.random.choice(range(0, nsamples), size=nsamples, replace=False)
dynamic = dynamic[new_train_idx]
static = static[new_train_idx]
labels = labels[new_train_idx]

# store data
print 'Storing data...'
np.save('/storage/hpc_anna/GMiC/Data/SpliceData/preprocessed/train_dynamic.npy', dynamic)
np.save('/storage/hpc_anna/GMiC/Data/SpliceData/preprocessed/train_static.npy', static)
np.save('/storage/hpc_anna/GMiC/Data/SpliceData/preprocessed/train_labels.npy', labels)
