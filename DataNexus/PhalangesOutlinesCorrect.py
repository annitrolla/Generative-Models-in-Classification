import numpy as np

# load dataset
print 'Loading data...'
train_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/raw/PhalangesOutlinesCorrect_TRAIN', delimiter=',')
test_raw = np.loadtxt('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/raw/PhalangesOutlinesCorrect_TEST', delimiter=',')
all_raw = np.concatenate((train_raw, test_raw))

# extract labels
print 'Transforming data...'
labels = all_raw[:, 0].astype(int);
dynamic = all_raw[:, 1:].reshape(2658, 1, 80)

# store data
print 'Storing data...'
np.save('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/preprocessed/train_dynamic.npy', dynamic)
np.save('/storage/hpc_anna/GMiC/Data/PhalangesOutlinesCorrect/preprocessed/train_labels.npy', labels)

