"""

The experiment combines the Fourier features and the features extraced from HMM generative model.
After this augmented dataset is created we train and evaluate Random Forest on it.

"""

import numpy as np 
from DataNexus.datahandler import DataHandler
from HMM.hmm_classifier import HMMClassifier 

# parameters
NSTATES = 20
NITERS = 100

# load the data
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

# split the training data into two halves
#   fh stands for first half
#   sh stands for second half
fh_data, fh_labels, sh_data, sh_labels = DataHandler.split(0.5, train_data, train_labels)

# train HMM on first 50% of the training set
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train_models(NSTATES, NITERS, fh_data, fh_labels)

# feed second 50% of the training set into the HMM to obtain
# pos/neg ratio for every sequence in the second half of the training set
ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, sh_data)

# augment second 50% of the training set with the ratios thus producing and enriched dataset
dh_sh = DataHandler(None)
dh_sh.train_data = sh_data
dh_sh.train_labels = sh_labels
dh_sh.test_data = np.array(dh.test_data, copy=True)
dh_sh.test_labels = np.array(dh.test_labels, copy=True)

# train RF on the enriched dataset

# test RF on the test set
