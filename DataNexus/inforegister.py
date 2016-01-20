"""

Preprocess Inforegister dataset

"""

import sys
import numpy as np
import pandas as pd

# load data
print 'Loading data...'
static = pd.read_csv('/storage/hpc_anna/GMiC/Data/Inforegister/raw/static.csv', sep=',', header=0)
dynamic = pd.read_csv('/storage/hpc_anna/GMiC/Data/Inforegister/raw/dynamic.csv', sep=',', header=0)
labels = pd.read_csv('/storage/hpc_anna/GMiC/Data/Inforegister/raw/labels.csv', sep=',', header=0)

# parameters
seqlen = 3

#
# static features
#
static = static.drop('case_name', 1)
train_static = np.array(static)

#
# dynamic features
#
print 'Encoding non-numerics...'
def encode_non_numeric(data, column):
    options = list(data[column].unique())
    for i, option in enumerate(options):
        data.loc[:, column] = data.loc[:, column].replace(option, i + 1)
    return data

dynamic = encode_non_numeric(dynamic, 'tax_declar')
dynamic = encode_non_numeric(dynamic, 'debt_decreased')
dynamic = encode_non_numeric(dynamic, 'event')

case_names = dynamic['case_name'].unique()

print 'Converting train dynamic features to 3D structure...'
n_samples = len(case_names)
train_dynamic = np.zeros((n_samples, dynamic.shape[1] - 1, seqlen))
for i, sid in enumerate(case_names):
    sys.stdout.write('{0}/{1}\r'.format(i, n_samples))
    sys.stdout.flush()
    session_data = dynamic[dynamic['case_name'] == sid].drop(['case_name'], axis=1)
    train_dynamic[i, :, :] = np.array(session_data).T

#
# labels
#
train_labels = np.ravel(np.array(labels))


#
# oversampling
#
positive_idx = np.where(train_labels == 1)[0]
n_positive = sum(train_labels)
n_negative = len(train_labels) - n_positive
oversample_idx = np.random.choice(positive_idx, n_negative - n_positive, replace=True)
train_static = np.concatenate((train_static, train_static[oversample_idx]))
train_dynamic = np.concatenate((train_dynamic, train_dynamic[oversample_idx]))
train_labels = np.concatenate((train_labels, train_labels[oversample_idx]))


#
# shuffle
#
print 'Shuffling...'
n_samples = len(train_labels)
new_idx = np.random.choice(range(0, n_samples), size=n_samples, replace=False)
train_static = train_static[new_idx]
train_dynamic = train_dynamic[new_idx]
train_labels = train_labels[new_idx]


#
# store the dataset
#
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_labels.npy', train_labels)

