"""

Preprocess Inforegister dataset
into statical and dynamical features and split training and test

"""

import sys
import numpy as np
import pandas as pd
import random

# parameters
seqlen = 5


# load data
print 'Loading data...'
data = pd.read_csv('/storage/hpc_anna/GMiC/Data/Inforegister/merged_inforegister.csv', sep=',', quotechar='"')

group_by_casename = data.groupby('case_name')


#
# static features
#

social_names = ['pr','deg','score','td','tdp','tdd','tdi','tdip','md','decl']
social_vars = []
for j,s in enumerate(social_names):
    social_vars += [s + "_" + str(i) for i in xrange(1,7)]
    
static_vars = ['case_name','status_code', 'age', 'label'] + social_vars
static = group_by_casename[static_vars].first()


#
# dynamic features
# 
dynamic_vars = ['case_name','date','state_tax_lq','lab_tax_lq','debt_sum','max_days_due','tax_declar','tax_debt','debt_balances','event','exp_payment','month']
dynamic = data[dynamic_vars]

session_length = group_by_casename['case_name'].count()

# keep only sessions that are longer than [seqlen] events
long_sessions = session_length[session_length >= seqlen].keys()
train_dynamic = dynamic[dynamic['case_name'].isin(long_sessions)]
train_static = static[static['case_name'].isin(long_sessions)]

# encode labels
train_static['label'] = (train_static['label']=='successful') * 1
train_labels = pd.DataFrame(train_static['label'])

# encode non-numeric data
print 'Encoding non-numerics...'
def encode_non_numeric(train, column):

    # compose full list of options
    options = list(train[column].unique())

    # encode them with integers
    for i, option in enumerate(options):
        train.loc[:, column] = train.loc[:, column].replace(option, i + 1)

    return train

train_dynamic = encode_non_numeric(train_dynamic, 'event')
train_dynamic = encode_non_numeric(train_dynamic, 'tax_declar')

# fill NAs
train_static = train_static.fillna(0)
train_dynamic = train_dynamic.fillna(0)

# prepare final 3D matrix
print 'Converting dynamic features to 3D structure...'
nsamples = len(train_static)
nfeatures = train_static.shape[1] - 2  # drop labels, case_name
nseqfeatures = train_dynamic.shape[1] - 2  # drop case_name, date
dynamic_all = np.zeros((nsamples, nseqfeatures, seqlen))

# put each sessions's data into the final matrix
for i, sid in enumerate(long_sessions):
    sys.stdout.write('{0}/{1}\r'.format(i, nsamples))
    sys.stdout.flush()
    session_data = train_dynamic[train_dynamic['case_name'] == sid].sort('date', ascending=1)
    session_data = session_data.iloc[:seqlen, :].drop(['case_name', 'date'], axis=1)
    dynamic_all[i, :, :] = np.array(session_data).T
train_dynamic = dynamic_all

# drop labels from static
train_static = train_static.drop('label', axis=1)
train_static = train_static.drop('case_name', axis=1)

train_static = np.array(train_static)
train_labels = np.array(train_labels)

# store the dataset
print 'Storing the dataset...'
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_static.npy', train_static)
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_dynamic.npy', train_dynamic)
np.save('/storage/hpc_anna/GMiC/Data/Inforegister/preprocessed/train_labels.npy', train_labels)

