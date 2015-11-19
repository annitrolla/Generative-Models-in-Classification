"""

Preprocess Plumbr dataset
into statical and dynamical features and split training and test

"""

import sys
import numpy as np
import pandas as pd
import random

# parameters
seqlen = 70


# load data
print 'Loading data...'
data = pd.read_csv('/storage/hpc_anna/GMiC/Data/Piper/raw/corrected_dt.txt', sep=' ', quotechar='"', header=None, names=['accountId','sessionId','javaVersion','cpuCoreCount','maxAvailableMemory','timestamp','duration','gcCause','gcAction','gcName','PermGenUsedBefore','PermGenMaxBefore','EdenSpaceUsedBefore','EdenSpaceMaxBefore','OldGenUsedBefore','OldGenMaxBefore','SurvivorSpaceUsedBefore','SurvivorSpaceMaxBefore','PermGenUsedAfter','PermGenMaxAfter','EdenSpaceUsedAfter','EdenSpaceMaxAfter','OldGenUsedAfter','OldGenMaxAfter','SurvivorSpaceUsedAfter','SurvivorSpaceMaxAfter','allocationRate','promotionRate','maturityRate'])

# find GC overhead for each session
print 'Computing GC overhead...'
group_by_sessionId = data.groupby('sessionId')
gc_overhead = group_by_sessionId['duration'].sum() / (group_by_sessionId['timestamp'].max() - group_by_sessionId['timestamp'].min())
gc_overhead = gc_overhead.replace([np.inf, -np.inf], 0.0)
gc_overhead = gc_overhead[gc_overhead <= 1.0]

# convert gc_overhead to DataFrame
gc_overhead = pd.DataFrame(gc_overhead)
gc_overhead.index.name = 'sessionId'
gc_overhead.reset_index(inplace=True)
gc_overhead.columns = ['sessionId', 'gcOverhead']

# set positive label if overhead is larger than 5%
gc_overhead['label'] = (gc_overhead['gcOverhead'] > 0.05) * 1


#
# static features
#
print 'Building static features...'
static = group_by_sessionId['sessionId', 'javaVersion', 'cpuCoreCount', 'maxAvailableMemory', 'PermGenMaxBefore', 'OldGenMaxBefore', 'PermGenMaxAfter', 'OldGenMaxAfter'].first()
static = static[static['sessionId'].isin(gc_overhead['sessionId'])]

# recode javaVersion
static.loc[:, 'javaVersion'] = static.loc[:, 'javaVersion'].replace(1.7, 1)
static.loc[:, 'javaVersion'] = static.loc[:, 'javaVersion'].replace(1.8, 2)

# replace -1 with 0 for some integer fetatures
static.loc[:, 'PermGenMaxBefore'] = static.loc[:, 'PermGenMaxBefore'].replace(-1, 0)
static.loc[:, 'PermGenMaxAfter'] = static.loc[:, 'PermGenMaxAfter'].replace(-1, 0)


#
# dynamic features
# 
print 'Building dynamic features...'
session_length = group_by_sessionId['sessionId'].count()
dynamic = data[data['sessionId'].isin(gc_overhead['sessionId'])]

# keep only sessions that are longer than [seqlen] events
long_sessions = session_length[session_length >= seqlen].keys()
gc_overhead = gc_overhead[gc_overhead['sessionId'].isin(long_sessions)]
dynamic = dynamic[dynamic['sessionId'].isin(long_sessions)]
static = static[static['sessionId'].isin(long_sessions)]

# drop columns we are not interested in
dynamic = dynamic.drop(['accountId', 'javaVersion', 'cpuCoreCount', 'maxAvailableMemory', 'PermGenMaxBefore',
                        'OldGenMaxBefore', 'PermGenMaxAfter', 'OldGenMaxAfter', 'gcName', 'allocationRate',
                        'promotionRate', 'maturityRate'], axis=1)

# keep only samples with positive GC duraion
dynamic = dynamic[dynamic['duration'] >= 0]

# encode gcCause
gc_causes = list(dynamic['gcCause'].unique())
for i, gc_cause in enumerate(gc_causes):
    dynamic.loc[:, 'gcCause'] = dynamic.loc[:, 'gcCause'].replace(gc_cause, i)

# encode gcAction
gc_actions = list(dynamic['gcAction'].unique())
for i, gc_action in enumerate(gc_actions):
        dynamic.loc[:, 'gcAction'] = dynamic.loc[:, 'gcAction'].replace(gc_action, i)

# balance the dataset
print 'Balancing the dataset...'
n_pos_class = np.sum(gc_overhead['label'] == 1)
pos_class_sessions = list(gc_overhead[gc_overhead['label'] == 1]['sessionId'])
neg_class_sessions = list(gc_overhead[gc_overhead['label'] == 0]['sessionId'])
neg_class_sessions = random.sample(neg_class_sessions, n_pos_class)  # take as many neg class sessions as many positive classes
take_sessions = pos_class_sessions + neg_class_sessions
take_sessions = sorted(take_sessions)

static = static[static['sessionId'].isin(take_sessions)]
static = static.sort('sessionId', ascending=1)
dynamic = dynamic[dynamic['sessionId'].isin(take_sessions)]
dynamic = dynamic.sort('sessionId', ascending=1)
gc_overhead = gc_overhead[gc_overhead['sessionId'].isin(take_sessions)]
gc_overhead = gc_overhead.sort('sessionId', ascending=1)

# prepare final 3D matrix
print 'Converting dynamic features to 3D structure...'
nsamples = len(take_sessions)
nfeatures = static.shape[1] - 1  # drop sessionId
nseqfeatures = dynamic.shape[1] - 2  # drop sessionId, timestamp
dynamic_all = np.zeros((nsamples, nseqfeatures, seqlen))

# put each sessions's data into the final matrix
for i, sid in enumerate(take_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, nsamples))
    sys.stdout.flush()
    session_data = dynamic[dynamic['sessionId'] == sid].sort('timestamp', ascending=1)
    session_data = session_data.iloc[:seqlen, :].drop(['sessionId', 'timestamp'], axis=1)
    dynamic_all[i, :, :] = np.array(session_data).T

# drop sessionId from static
static = static.drop('sessionId', axis=1)

# convert static and gc_overhead to numpy structures
static_all = np.array(static)
labels_all = np.array(gc_overhead['label'])

# split training and test
print 'Splitting training and test sets...'
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

train_static = static_all[train_idx, :]
train_dynamic = dynamic_all[train_idx, :, :]
test_static = static_all[test_idx, :]
test_dynamic = dynamic_all[test_idx, :, :]
train_labels = labels_all[train_idx]
test_labels = labels_all[test_idx]

# store the dataset
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_static.npy', test_static)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_dynamic.npy', test_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/train_labels.npy', train_labels)
np.save('/storage/hpc_anna/GMiC/Data/Piper/preprocessed/test_labels.npy', test_labels)

