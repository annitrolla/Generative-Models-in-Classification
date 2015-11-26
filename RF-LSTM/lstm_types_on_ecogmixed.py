"""

Try different LSTM architectures and different enrichment schemes on synthetic data
  - Generative LSTM, enrichment with ratios
  - Generative LSTM, enrichment with LSTM layer activations
  - Discriminative LSTM, enrichment with ratios
  - Discriminative LSTM, enrichment with LSTM layer activations

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier, LSTMDiscriminative


#
# Parameters
#
nfolds = 5
nestimators = 300

g_lstmsize = 2000
g_lstmdropout = 0.5
g_lstmoptim = 'adagrad'
g_lstmnepochs = 20
g_lstmbatch = 64

d_lstmsize = 300
d_fcsize = 100
d_lstmdropout = 0.8
d_lstmoptim = 'adadelta'
d_lstmnepochs = 20
d_lstmbatch = 128


#
# Load the dataset
#
print 'Loading the dataset..'
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')


#
# Merge train and test
#
print "Combining data from train and test for CV predictions..."
static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]


#
# Cross-validation to collect enrichment features
#
ratios_generative = np.empty(len(labels_all))
ratios_discriminative = np.empty(len(labels_all))
activations_generative = np.empty((len(labels_all), g_lstmsize * 2)) # LSTM size in generative case
activations_discriminative = np.empty((len(labels_all), 100)) # FC layer in disciminative case

predict_idx_list = np.array_split(range(nsamples), nfolds)
for fid, predict_idx in enumerate(predict_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(predict_idx))
 
   
    #
    # Generative LSTM
    #
    print "    Extracting ratios and activations from generative LSTM..."

    # train the models
    lstmcl = LSTMClassifier(g_lstmsize, g_lstmdropout, g_lstmoptim, g_lstmnepochs, g_lstmbatch)
    model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
    
    # extract ratios
    mse_pos, mse_neg = lstmcl.predict_mse(model_pos, model_neg, dynamic_all[predict_idx]) 
    ratios_generative[predict_idx] = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[predict_idx])

    # extract activations
    activations_pos = lstmcl.activations(model_pos, dynamic_all[predict_idx])
    activations_neg = lstmcl.activations(model_neg, dynamic_all[predict_idx])
    activations_generative[predict_idx] = np.concatenate((activations_pos[:, -1, :], activations_neg[:, -1, :]), axis=1)


    #
    # Discriminative LSTM
    #
    print "    Extracting ratios and activations from discriminative LSTM..."

    # train the model
    lstmcl = LSTMDiscriminative(d_lstmsize, d_fcsize, d_lstmdropout, d_lstmoptim, d_lstmnepochs, d_lstmbatch)
    model = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])

    # extract ratios
    ratios_discriminative[predict_idx] = lstmcl.pos_neg_ratios(model, dynamic_all[predict_idx])

    # extract activations
    activations_discriminative[predict_idx] = lstmcl.activations(model, dynamic_all[predict_idx]) 


#
# Prepare combined datasets for the future experiments
#
print 'Enriching the datasets...'
enriched_by_generative_ratios = np.concatenate((static_all, np.matrix(ratios_generative).T), axis=1)
enriched_by_generative_activations = np.concatenate((static_all, activations_generative), axis=1)
enriched_by_discriminative_ratios = np.concatenate((static_all, np.matrix(ratios_discriminative).T), axis=1)
enriched_by_discriminative_activations = np.concatenate((static_all, activations_discriminative), axis=1)


#
# Split the data into training and test
#
train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))


#
# Train the models and report their performance
#
print 'Training the models...'

# RF on static data enriched by generative ratios
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(enriched_by_generative_ratios[train_idx], labels_all[train_idx])
print "===> RF on static enriched by generative ratios: %.4f" % rf.score(enriched_by_generative_ratios[test_idx], labels_all[test_idx])

# RF on static data enriched by generative activations
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(enriched_by_generative_activations[train_idx], labels_all[train_idx])
print "===> RF on static enriched by generative activations: %.4f" % rf.score(enriched_by_generative_activations[test_idx], labels_all[test_idx])

# RF on static data enriched by discriminative ratios
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(enriched_by_discriminative_ratios[train_idx], labels_all[train_idx])
print "===> RF on static enriched by discriminative ratios: %.4f" % rf.score(enriched_by_discriminative_ratios[test_idx], labels_all[test_idx])

# RF on static data enriched by discriminative activations
rf = RandomForestClassifier(n_estimators=nestimators)
rf.fit(enriched_by_discriminative_activations[train_idx], labels_all[train_idx])
print "===> RF on static enriched by discriminative activations: %.4f" % rf.score(enriched_by_discriminative_activations[test_idx], labels_all[test_idx])

