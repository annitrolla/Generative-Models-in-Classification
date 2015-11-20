import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
import sys
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

# division of static features on numeric...:
train_static_numeric = train_static[['sequence_nr', 'Age']]
test_static_numeric = test_static[['sequence_nr', 'Age']]

#...and nonnumeric
train_static_nonnumeric = train_static[['sequence_nr', 'Diagnosis', 'Diagnosis code', 'Diagnosis Treatment Combination ID', 'Treatment code']]
test_static_nonnumeric = test_static[['sequence_nr', 'Diagnosis', 'Diagnosis code', 'Diagnosis Treatment Combination ID', 'Treatment code']]

# division of dynamic features on numeric:
train_dynamic_numeric = train_dynamic[['sequence_nr', 'Number of executions']]
test_dynamic_numeric = test_dynamic[['sequence_nr', 'Number of executions']]

#..and nonnumeric
train_dynamic_nonnumeric = train_dynamic[['sequence_nr', 'activity_name', 'Activity code', 'group', 'Producer code', 'Section', 'Specialism code']]
test_dynamic_nonnumeric = test_dynamic[['sequence_nr', 'activity_name', 'Activity code', 'group', 'Producer code', 'Section', 'Specialism code']]

# one-hot encoding for static non-numeric and dynamic non-numeric
# encode non-numeric data
print 'Encoding non-numerics...'
def encode_one_hot(train, test, column):

    # compose full list of options
    options = list(set(list(train[column].unique()) + list(test[column].unique())))

    # encode them with integers
    for i, option in enumerate(options):
        train.loc[:, column] = train.loc[:, column].replace(option, i + 1)
        test.loc[:, column] = test.loc[:, column].replace(option, i + 1)

    # recode into one-hot vectors
    options = list(set(list(train[column].unique()) + list(test[column].unique())))
    enc = OneHotEncoder(sparse=False)
    enc.fit(np.matrix(options).T)
    original_names = dict((i, a) for i, a in enumerate(train.columns.values))
    train = pd.concat([train, pd.DataFrame(enc.transform(np.matrix(train[column]).T))], axis=1, ignore_index=True)
    test = pd.concat([test, pd.DataFrame(enc.transform(np.matrix(test[column]).T))], axis=1, ignore_index=True)
    train = train.rename(columns=original_names)
    test = test.rename(columns=original_names)

    # drop the original of the encoded column
    train = train.drop(column, axis=1)
    test = test.drop(column, axis=1)

    return train, test

def encode_as_int(train, test, column):

    # compose full list of options
    options = list(set(list(train[column].unique()) + list(test[column].unique())))

    # encode them with integers
    for i, option in enumerate(options):
        train.loc[:, column] = train.loc[:, column].replace(option, i + 1)
        test.loc[:, column] = test.loc[:, column].replace(option, i + 1)

    return train, test


train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'activity_name')
train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'Activity code')
train_static_nonnumeric, test_static_nonnumeric = encode_one_hot(train_static_nonnumeric, test_static_nonnumeric, 'Diagnosis')
train_static_nonnumeric, test_static_nonnumeric = encode_one_hot(train_static_nonnumeric, test_static_nonnumeric, 'Diagnosis code')
train_static_nonnumeric, test_static_nonnumeric = encode_one_hot(train_static_nonnumeric, test_static_nonnumeric, 'Diagnosis Treatment Combination ID')
train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'group')
train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'Producer code')
train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'Section')
train_dynamic_nonnumeric, test_dynamic_nonnumeric = encode_as_int(train_dynamic_nonnumeric, test_dynamic_nonnumeric, 'Specialism code')
train_static_nonnumeric, test_static_nonnumeric = encode_one_hot(train_static_nonnumeric, test_static_nonnumeric, 'Treatment code')

# fill NAs
train_static_nonnumeric = train_static_nonnumeric.fillna(0)
train_dynamic_nonnumeric = train_dynamic_nonnumeric.fillna(0)
test_static_nonnumeric = test_static_nonnumeric.fillna(0)
test_dynamic_nonnumeric = test_dynamic_nonnumeric.fillna(0)
train_static_numeric = train_static_numeric.fillna(0)
train_dynamic_numeric = train_dynamic_numeric.fillna(0)
test_static_numeric = test_static_numeric.fillna(0)
test_dynamic_numeric = test_dynamic_numeric.fillna(0)

# select session with sequences of length at least [seqlen]
print 'Dropping short sequencies...'
train_session_length = train_group_by_sequence_nr['sequence_nr'].count()
test_session_length = test_group_by_sequence_nr['sequence_nr'].count()
train_take_sessions = train_session_length[train_session_length >= seqlen].keys()
test_take_sessions = test_session_length[test_session_length >= seqlen].keys()

train_static_numeric = train_static_numeric[train_static_numeric['sequence_nr'].isin(train_take_sessions)]
train_static_nonnumeric = train_static_nonnumeric[train_static_nonnumeric['sequence_nr'].isin(train_take_sessions)]

train_dynamic_nonnumeric = train_dynamic_nonnumeric[train_dynamic_nonnumeric['sequence_nr'].isin(train_take_sessions)]
train_dynamic_numeric = train_dynamic_numeric[train_dynamic_numeric['sequence_nr'].isin(train_take_sessions)]

train_labels = pd.DataFrame(train_labels)
train_labels.index.name = 'sequence_nr'
train_labels.reset_index(inplace=True)
train_labels.columns = ['sequence_nr', 'label']
train_labels = train_labels[train_labels['sequence_nr'].isin(train_take_sessions)]

test_static_numeric = test_static_numeric[test_static_numeric['sequence_nr'].isin(test_take_sessions)]
test_static_nonnumeric = test_static_nonnumeric[test_static_nonnumeric['sequence_nr'].isin(test_take_sessions)]
test_dynamic_numeric = test_dynamic_numeric[test_dynamic_numeric['sequence_nr'].isin(test_take_sessions)]
test_dynamic_nonnumeric = test_dynamic_nonnumeric[test_dynamic_nonnumeric['sequence_nr'].isin(test_take_sessions)]


test_labels = pd.DataFrame(test_labels)
test_labels.index.name = 'sequence_nr'
test_labels.reset_index(inplace=True)
test_labels.columns = ['sequence_nr', 'label']
test_labels = test_labels[test_labels['sequence_nr'].isin(test_take_sessions)]

# convert dynamic to numpy
print 'Converting train dynamic features to 3D structure...'
n_train = len(train_labels)
train_dynamic_numeric_np = np.zeros((n_train, train_dynamic_numeric.shape[1] - 1, seqlen))
train_dynamic_nonnumeric_np = np.zeros((n_train, train_dynamic_nonnumeric.shape[1] - 1, seqlen), dtype='int')

for i, sid in enumerate(train_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_train))
    sys.stdout.flush()
    session_data_numeric = train_dynamic_numeric[train_dynamic_numeric['sequence_nr'] == sid]
    session_data_numeric = session_data_numeric.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    train_dynamic_numeric_np[i, :, :] = np.array(session_data_numeric).T

    session_data_nonnumeric = train_dynamic_nonnumeric[train_dynamic_nonnumeric['sequence_nr'] == sid]
    session_data_nonnumeric = session_data_nonnumeric.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    train_dynamic_nonnumeric_np[i, :, :] = np.array(session_data_nonnumeric).T


print 'Converting test dynamic features to 3D structure...'
n_test = len(test_labels)
test_dynamic_numeric_np = np.zeros((n_test, test_dynamic_numeric.shape[1] - 1, seqlen))
test_dynamic_nonnumeric_np = np.zeros((n_test, test_dynamic_nonnumeric.shape[1] - 1, seqlen), dtype='int')

for i, sid in enumerate(test_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_test))
    sys.stdout.flush()
    session_data_numeric = test_dynamic_numeric[test_dynamic_numeric['sequence_nr'] == sid]
    session_data_numeric = session_data_numeric.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    test_dynamic_numeric_np[i, :, :] = np.array(session_data_numeric).T

    session_data_nonnumeric = test_dynamic_nonnumeric[test_dynamic_nonnumeric['sequence_nr'] == sid]
    session_data_nonnumeric = session_data_nonnumeric.iloc[:seqlen, :].drop(['sequence_nr'], axis=1)
    test_dynamic_nonnumeric_np[i, :, :] = np.array(session_data_nonnumeric).T


# drop sequence_nr
train_static_numeric = train_static_numeric.drop('sequence_nr', axis=1)
train_static_nonnumeric = train_static_nonnumeric.drop('sequence_nr', axis=1)
train_dynamic_numeric = train_dynamic_numeric.drop('sequence_nr', axis=1)
train_dynamic_nonnumeric = train_dynamic_nonnumeric.drop('sequence_nr', axis=1)
train_labels = train_labels.drop('sequence_nr', axis=1)

test_static_numeric = test_static_numeric.drop('sequence_nr', axis=1)
test_static_nonnumeric = test_static_nonnumeric.drop('sequence_nr', axis=1)
test_dynamic_numeric = test_dynamic_numeric.drop('sequence_nr', axis=1)
test_dynamic_nonnumeric = test_dynamic_nonnumeric.drop('sequence_nr', axis=1)
test_labels = test_labels.drop('sequence_nr', axis=1)

# put into numpy matrices
train_static_numeric = np.array(train_static_numeric)
train_static_nonnumeric = np.array(train_static_nonnumeric, dtype='int')
train_dynamic_numeric = train_dynamic_numeric_np
train_dynamic_nonnumeric = train_dynamic_nonnumeric_np
train_labels = np.array(train_labels.T)[0]

test_static_numeric = np.array(test_static_numeric)
test_static_nonnumeric = np.array(test_static_nonnumeric, dtype='int')
test_dynamic_numeric = test_dynamic_numeric_np
test_dynamic_nonnumeric = test_dynamic_nonnumeric_np
test_labels = np.array(test_labels.T)[0]

print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_static_numeric.npy' % dataset, train_static_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_static_nonnumeric.npy' % dataset, train_static_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_dynamic_numeric.npy' % dataset, train_dynamic_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_dynamic_nonnumeric.npy' % dataset, train_dynamic_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_static_numeric.npy' % dataset, test_static_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_static_nonnumeric.npy' % dataset, test_static_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_dynamic_numeric.npy' % dataset, test_dynamic_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_dynamic_nonnumeric.npy' % dataset, test_dynamic_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_labels.npy' % dataset, train_labels)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_labels.npy' % dataset, test_labels)
