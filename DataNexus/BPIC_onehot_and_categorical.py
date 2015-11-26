import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier
import sys
import pandas as pd
import argparse
import time
import datetime
from sklearn.preprocessing import OneHotEncoder
import cPickle

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

# one-hot encoding for static non-numeric and dynamic non-numeric
# encode non-numeric data
def encode_one_hot(train, test, column):

    # compose full list of options
    options = list(set(list(train[column].unique()) + list(test[column].unique())))

    # encode them with integers
    for i, option in enumerate(options):
        #train.loc[:, column] = train.loc[:, column].replace(option, i + 1)
        #test.loc[:, column] = test.loc[:, column].replace(option, i + 1)
        train.loc[train[column] == option, column] = i
        test.loc[test[column] == option, column] = i

    # recode into one-hot vectors
    options = list(set(list(train[column].unique()) + list(test[column].unique())))
    enc = OneHotEncoder(sparse=False)
    enc.fit(np.matrix(options).T)
    original_names = dict((i, a) for i, a in enumerate(train.columns.values))
    
    oh = pd.DataFrame(enc.transform(np.matrix(train[column]).T))
    oh = oh.set_index(train.index.values)
    train = pd.concat([train, oh], axis=1, ignore_index=True)
    
    oh = pd.DataFrame(enc.transform(np.matrix(test[column]).T))
    oh = oh.set_index(test.index.values)
    test = pd.concat([test, oh], axis=1, ignore_index=True)
    
    train = train.rename(columns=original_names)
    test = test.rename(columns=original_names)

    # drop the original of the encoded column
    train = train.drop(column, axis=1)
    test = test.drop(column, axis=1)

    return train, test

def encode_as_int(train, test, column):

    # compose full list of options
    options = train[column].unique()
    test_only_options = list(set(test[column].unique()) - set(options))
    
    # use last option to designate "unknown"
    unknown_id = len(options)

    # encode them with integers
    mapping = {}
    for i, option in enumerate(options):
        train.loc[:, column] = train.loc[:, column].replace(option, i)
        test.loc[:, column] = test.loc[:, column].replace(option, i)
        mapping[option] = i

    for option in test_only_options:
        test.loc[:, column] = test.loc[:, column].replace(option, unknown_id)
        mapping[option] = unknown_id

    return train, test, mapping

# cutting sequences up to pre-defined length
print "Cutting sequenes up to %d..." % seqlen
n_train = len(train_labels) 
train_dynamic_numeric_short = pd.DataFrame(columns=train_dynamic_numeric.columns)
train_dynamic_nonnumeric_short = pd.DataFrame(columns=train_dynamic_nonnumeric.columns)
for i, sid in enumerate(train_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_train))
    sys.stdout.flush()
    session_data_numeric = train_dynamic_numeric[train_dynamic_numeric['sequence_nr'] == sid]
    session_data_numeric = session_data_numeric.iloc[:seqlen, :]
    train_dynamic_numeric_short = train_dynamic_numeric_short.append(session_data_numeric)
    
    session_data_nonnumeric = train_dynamic_nonnumeric[train_dynamic_nonnumeric['sequence_nr'] == sid]
    session_data_nonnumeric = session_data_nonnumeric.iloc[:seqlen, :]
    train_dynamic_nonnumeric_short = train_dynamic_nonnumeric_short.append(session_data_nonnumeric)

train_dynamic_numeric = train_dynamic_numeric_short
train_dynamic_nonnumeric = train_dynamic_nonnumeric_short 

n_test = len(test_labels) 
test_dynamic_numeric_short = pd.DataFrame(columns=test_dynamic_numeric.columns)
test_dynamic_nonnumeric_short = pd.DataFrame(columns=test_dynamic_nonnumeric.columns)
for i, sid in enumerate(test_take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, n_test))
    sys.stdout.flush()
    session_data_numeric = test_dynamic_numeric[test_dynamic_numeric['sequence_nr'] == sid]
    session_data_numeric = session_data_numeric.iloc[:seqlen, :]
    test_dynamic_numeric_short = test_dynamic_numeric_short.append(session_data_numeric)
    
    session_data_nonnumeric = test_dynamic_nonnumeric[test_dynamic_nonnumeric['sequence_nr'] == sid]
    session_data_nonnumeric = session_data_nonnumeric.iloc[:seqlen, :]
    test_dynamic_nonnumeric_short = test_dynamic_nonnumeric_short.append(session_data_nonnumeric)

test_dynamic_numeric = test_dynamic_numeric_short
test_dynamic_nonnumeric = test_dynamic_nonnumeric_short 

# pos/neg
train_pos_sessions = list(train_labels[train_labels['label']==1]['sequence_nr'])
train_neg_sessions = list(train_labels[train_labels['label']==0]['sequence_nr'])
test_pos_sessions = list(test_labels[test_labels['label']==1]['sequence_nr'])
test_neg_sessions = list(test_labels[test_labels['label']==0]['sequence_nr'])

train_dynamic_nonnumeric_pos = train_dynamic_nonnumeric[train_dynamic_nonnumeric['sequence_nr'].isin(train_pos_sessions)]
train_dynamic_nonnumeric_neg = train_dynamic_nonnumeric[train_dynamic_nonnumeric['sequence_nr'].isin(train_neg_sessions)]
test_dynamic_nonnumeric_pos = test_dynamic_nonnumeric[test_dynamic_nonnumeric['sequence_nr'].isin(test_pos_sessions)]
test_dynamic_nonnumeric_neg = test_dynamic_nonnumeric[test_dynamic_nonnumeric['sequence_nr'].isin(test_neg_sessions)]

print 'Encoding non-numerics...'
mappings = {}
mappings['pos'] = {}
mappings['neg'] = {}

# dynamic for multinomial
for cid, column in enumerate(['activity_name', 'Activity code', 'group', 'Producer code', 'Section', 'Specialism code']):
    print '    encoding %s' % column
    train_dynamic_nonnumeric_pos, test_dynamic_nonnumeric_pos, mapping = encode_as_int(train_dynamic_nonnumeric_pos, test_dynamic_nonnumeric_pos, column) 
    mappings['pos'][cid] = mapping
    train_dynamic_nonnumeric_neg, test_dynamic_nonnumeric_neg, mapping = encode_as_int(train_dynamic_nonnumeric_neg, test_dynamic_nonnumeric_neg, column)
    mappings['neg'][cid] = mapping

# convert the mapping into pos-to-neg and neg-to-pos mappings
pos_to_neg = {}
for cid, column in enumerate(['activity_name', 'Activity code', 'group', 'Producer code', 'Section', 'Specialism code']):

    # initialize resulting code with negative unknown
    pos_to_neg[cid] = np.ones(len(mappings['pos'][cid]), dtype='int') * max(mappings['neg'][cid].values())

    for name, p in mappings['pos'][cid].iteritems():
        n = mappings['neg'][cid].get(name, None)
        if n is not None:
            pos_to_neg[cid][p] = n

neg_to_pos = {}
for cid, column in enumerate(['activity_name', 'Activity code', 'group', 'Producer code', 'Section', 'Specialism code']):

    # initialize resulting code with negative unknown
    neg_to_pos[cid] = np.ones(len(mappings['neg'][cid]), dtype='int') * max(mappings['pos'][cid].values())

    for name, n in mappings['neg'][cid].iteritems():
        p = mappings['pos'][cid].get(name, None)
        if p is not None:
            neg_to_pos[cid][n] = p


# static
for column in ['Diagnosis', 'Diagnosis code', 'Diagnosis Treatment Combination ID', 'Treatment code']:
    print '    encoding %s' % column
    train_static_nonnumeric, test_static_nonnumeric = encode_one_hot(train_static_nonnumeric, test_static_nonnumeric, column)


#
# Convert Pandas data frames to numpy matrices
#
print 'Converting dynamic features to 3D structure...'

# dynamic numeric train
train_dynamic_numeric_np = np.zeros((n_train, train_dynamic_numeric.shape[1] - 1, seqlen))
for i, sid in enumerate(train_take_sessions):
    session_data_numeric = train_dynamic_numeric[train_dynamic_numeric['sequence_nr'] == sid]
    train_dynamic_numeric_np[i, :, :] = np.array(session_data_numeric.drop('sequence_nr', axis=1)).T

# dynamic nonnumeric train
n_train_pos = len(train_pos_sessions)
train_dynamic_nonnumeric_pos_np = np.zeros((n_train_pos, train_dynamic_nonnumeric_pos.shape[1] - 1, seqlen), dtype='int')
for i, sid in enumerate(train_pos_sessions):
    session_data = train_dynamic_nonnumeric_pos[train_dynamic_nonnumeric_pos['sequence_nr'] == sid]
    train_dynamic_nonnumeric_pos_np[i, :, :] = np.array(session_data.drop('sequence_nr', axis=1)).T

n_train_neg = len(train_neg_sessions)
train_dynamic_nonnumeric_neg_np = np.zeros((n_train_neg, train_dynamic_nonnumeric_neg.shape[1] - 1, seqlen), dtype='int')
for i, sid in enumerate(train_neg_sessions):
    session_data = train_dynamic_nonnumeric_neg[train_dynamic_nonnumeric_neg['sequence_nr'] == sid]
    train_dynamic_nonnumeric_neg_np[i, :, :] = np.array(session_data.drop('sequence_nr', axis=1)).T

# dynamic numeric test
test_dynamic_numeric_np = np.zeros((n_test, test_dynamic_numeric.shape[1] - 1, seqlen))
for i, sid in enumerate(test_take_sessions):
    session_data_numeric = test_dynamic_numeric[test_dynamic_numeric['sequence_nr'] == sid]
    test_dynamic_numeric_np[i, :, :] = np.array(session_data_numeric.drop('sequence_nr', axis=1)).T

# dynamic nonumeric test
n_test_pos = len(test_pos_sessions)
test_dynamic_nonnumeric_pos_np = np.zeros((n_test_pos, test_dynamic_nonnumeric_pos.shape[1] - 1, seqlen), dtype='int')
for i, sid in enumerate(test_pos_sessions):
    session_data = test_dynamic_nonnumeric_pos[test_dynamic_nonnumeric_pos['sequence_nr'] == sid]
    test_dynamic_nonnumeric_pos_np[i, :, :] = np.array(session_data.drop('sequence_nr', axis=1)).T

n_test_neg = len(test_neg_sessions)
test_dynamic_nonnumeric_neg_np = np.zeros((n_test_neg, test_dynamic_nonnumeric_neg.shape[1] - 1, seqlen), dtype='int')
for i, sid in enumerate(test_neg_sessions):
    session_data = test_dynamic_nonnumeric_neg[test_dynamic_nonnumeric_neg['sequence_nr'] == sid]
    test_dynamic_nonnumeric_neg_np[i, :, :] = np.array(session_data.drop('sequence_nr', axis=1)).T

# drop sequence_nr
train_static_numeric.drop('sequence_nr', axis=1, inplace=True)
train_static_nonnumeric.drop('sequence_nr', axis=1, inplace=True)
train_dynamic_numeric.drop('sequence_nr', axis=1, inplace=True)
train_dynamic_nonnumeric.drop('sequence_nr', axis=1, inplace=True)
train_dynamic_nonnumeric_pos.drop('sequence_nr', axis=1, inplace=True)
train_dynamic_nonnumeric_neg.drop('sequence_nr', axis=1, inplace=True)
train_labels.drop('sequence_nr', axis=1, inplace=True)

test_static_numeric.drop('sequence_nr', axis=1, inplace=True)
test_static_nonnumeric.drop('sequence_nr', axis=1, inplace=True)
test_dynamic_numeric.drop('sequence_nr', axis=1, inplace=True)
test_dynamic_nonnumeric.drop('sequence_nr', axis=1, inplace=True)
test_dynamic_nonnumeric_pos.drop('sequence_nr', axis=1, inplace=True)
test_dynamic_nonnumeric_neg.drop('sequence_nr', axis=1, inplace=True)
test_labels.drop('sequence_nr', axis=1, inplace=True)

# put into numpy matrices
train_static_numeric = np.array(train_static_numeric)
train_static_nonnumeric = np.array(train_static_nonnumeric, dtype='int')
train_dynamic_numeric = train_dynamic_numeric_np
train_dynamic_nonnumeric_pos = train_dynamic_nonnumeric_pos_np
train_dynamic_nonnumeric_neg = train_dynamic_nonnumeric_neg_np
train_labels = np.array(train_labels.T)[0]

test_static_numeric = np.array(test_static_numeric)
test_static_nonnumeric = np.array(test_static_nonnumeric, dtype='int')
test_dynamic_numeric = test_dynamic_numeric_np
test_dynamic_nonnumeric_pos = test_dynamic_nonnumeric_pos_np
test_dynamic_nonnumeric_neg = test_dynamic_nonnumeric_neg_np
test_labels = np.array(test_labels.T)[0]

print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_static_numeric.npy' % dataset, train_static_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_static_nonnumeric.npy' % dataset, train_static_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_dynamic_numeric.npy' % dataset, train_dynamic_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_dynamic_nonnumeric_pos.npy' % dataset, train_dynamic_nonnumeric_pos)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_dynamic_nonnumeric_neg.npy' % dataset, train_dynamic_nonnumeric_neg)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_static_numeric.npy' % dataset, test_static_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_static_nonnumeric.npy' % dataset, test_static_nonnumeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_dynamic_numeric.npy' % dataset, test_dynamic_numeric)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_dynamic_nonnumeric_pos.npy' % dataset, test_dynamic_nonnumeric_pos)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_dynamic_nonnumeric_neg.npy' % dataset, test_dynamic_nonnumeric_neg)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/train_labels.npy' % dataset, train_labels)
np.save('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/test_labels.npy' % dataset, test_labels)

print 'Storing mappings between test and train...'
with open('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/pos_to_neg.pkl' % dataset, 'w') as f:
    cPickle.dump(pos_to_neg, f)
with open('/storage/hpc_anna/GMiC/Data/BPIChallenge/%svar/preprocessed/neg_to_pos.pkl' % dataset, 'w') as f:
    cPickle.dump(neg_to_pos, f)





