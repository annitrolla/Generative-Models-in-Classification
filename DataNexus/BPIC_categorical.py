"""

Convert BPI Challenge data (4 datasets with different labelings) into format suitable
for out pipeline

"""

import sys
import numpy as np
import pandas as pd
import argparse
import time
import datetime
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='Convert BPIC2011 dataset for our pipeline')
parser.add_argument('-l', '--seqlen', dest='seqlen', type=int, required=True, help='Length of the sequence (shorter dropped, longer truncated)')
parser.add_argument('-d', '--dataset', dest='dataset', type=str, required=True, help='f1, f2, f3 or f4')
args = parser.parse_args()
seqlen = int(args.seqlen)
dataset = str(args.dataset)

# load the data
print 'Loading data...'
train = pd.read_csv('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/csv/train.csv' % dataset, sep=' ', quotechar='"')
test = pd.read_csv('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/csv/test.csv' % dataset, sep=' ', quotechar='"')
train = train.rename(columns={'time:timestamp': 'timestamp'})
test = test.rename(columns={'time:timestamp': 'timestamp'})

# split into static, dynamic and labels
train_group_by_sequence_nr = train.groupby('sequence_nr')
test_group_by_sequence_nr = test.groupby('sequence_nr')

# static features
print 'Producing static features...'
train_static = train_group_by_sequence_nr['sequence_nr', 'Age', 'Diagnosis', 'Diagnosis code', 'Diagnosis Treatment Combination ID', 'Treatment code'].first()
test_static = test_group_by_sequence_nr['sequence_nr', 'Age', 'Diagnosis', 'Diagnosis code', 'Diagnosis Treatment Combination ID', 'Treatment code'].first()

# dynamic features
print 'Producing dynamic features...'
train_dynamic = train[['sequence_nr', 'activity_name', 'Activity code', 'group', 'Number of executions', 'Producer code', 'Section', 'Specialism code']]
test_dynamic = test[['sequence_nr', 'activity_name', 'Activity code', 'group', 'Number of executions', 'Producer code', 'Section', 'Specialism code']]

# labels
train['label'] = train['label'] * 1
test['label'] = test['label'] * 1
train_labels = train_group_by_sequence_nr['label'].first()
test_labels = test_group_by_sequence_nr['label'].first()

# encode non-numeric data
print 'Encoding non-numerics...'
def encode_non_numeric(train, test, column):

    # compose full list of options
    options = list(set(list(train[column].unique()) + list(test[column].unique())))

    # encode them with integers
    for i, option in enumerate(options):
        train.loc[:, column] = train.loc[:, column].replace(option, i + 1)
        test.loc[:, column] = test.loc[:, column].replace(option, i + 1)

    return train, test

train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'activity_name')
train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'Activity code')
train_static, test_static = encode_non_numeric(train_static, test_static, 'Diagnosis')
train_static, test_static = encode_non_numeric(train_static, test_static, 'Diagnosis code')
train_static, test_static = encode_non_numeric(train_static, test_static, 'Diagnosis Treatment Combination ID')
train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'group')
train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'Producer code')
train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'Section')
train_dynamic, test_dynamic = encode_non_numeric(train_dynamic, test_dynamic, 'Specialism code')
train_static, test_static = encode_non_numeric(train_static, test_static, 'Treatment code')

# fill NAs
train_static = train_static.fillna(0)
train_dynamic = train_dynamic.fillna(0)
test_static = test_static.fillna(0)
test_dynamic = test_dynamic.fillna(0)

# select session with sequences of length at least [seqlen]
print 'Dropping short sequencies...'
train_session_length = train_group_by_sequence_nr['sequence_nr'].count()
test_session_length = test_group_by_sequence_nr['sequence_nr'].count()
train_take_sessions = train_session_length[train_session_length >= seqlen].keys()
test_take_sessions = test_session_length[test_session_length >= seqlen].keys()

train_static = train_static[train_static['sequence_nr'].isin(train_take_sessions)]
train_dynamic = train_dynamic[train_dynamic['sequence_nr'].isin(train_take_sessions)]
train_labels = pd.DataFrame(train_labels)
train_labels.index.name = 'sequence_nr'
train_labels.reset_index(inplace=True)
train_labels.columns = ['sequence_nr', 'label']
train_labels = train_labels[train_labels['sequence_nr'].isin(train_take_sessions)]

test_static = test_static[test_static['sequence_nr'].isin(test_take_sessions)]
test_dynamic = test_dynamic[test_dynamic['sequence_nr'].isin(test_take_sessions)]
test_labels = pd.DataFrame(test_labels)
test_labels.index.name = 'sequence_nr'
test_labels.reset_index(inplace=True)
test_labels.columns = ['sequence_nr', 'label']
test_labels = test_labels[test_labels['sequence_nr'].isin(test_take_sessions)]

# convert dynamic to numpy
print 'Converting train dynamic features to 3D structure...'
n_train = len(train_labels)
train_dynamic_np = np.zeros((n_train, train_dynamic.shape[1] - 1, seqlen))
for i, sid in enumerate(train_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_train))
    sys.stdout.flush()
    session_data = train_dynamic[train_dynamic['sequence_nr'] == sid]
    session_data = session_data.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    train_dynamic_np[i, :, :] = np.array(session_data).T

print 'Converting test dynamic features to 3D structure...'
n_test = len(test_labels)
test_dynamic_np = np.zeros((n_test, test_dynamic.shape[1] - 1, seqlen))
for i, sid in enumerate(test_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_test))
    sys.stdout.flush()
    session_data = test_dynamic[test_dynamic['sequence_nr'] == sid]
    session_data = session_data.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    test_dynamic_np[i, :, :] = np.array(session_data).T

# drop sequence_nr
train_static = train_static.drop('sequence_nr', axis=1)
train_dynamic = train_dynamic.drop('sequence_nr', axis=1)
train_labels = train_labels.drop('sequence_nr', axis=1)
test_static = test_static.drop('sequence_nr', axis=1)
test_dynamic = test_dynamic.drop('sequence_nr', axis=1)
test_labels = test_labels.drop('sequence_nr', axis=1)

# put into numpy matrices
train_static = np.array(train_static)
train_dynamic = train_dynamic_np
train_labels = np.array(train_labels.T)[0]
test_static = np.array(test_static)
test_dynamic = test_dynamic_np
test_labels = np.array(test_labels.T)[0]

# store the data
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_static.npy' % dataset, train_static)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_dynamic.npy' % dataset, train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_static.npy' % dataset, test_static)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_dynamic.npy' % dataset, test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/train_labels.npy' % dataset, train_labels)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%s/preprocessed/test_labels.npy' % dataset, test_labels)
