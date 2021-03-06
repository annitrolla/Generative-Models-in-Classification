"""

The experiment combines the Fourier features and the features extraced from HMM generative model.
After this augmented dataset is created we train and evaluate Random Forest on it.

"""

import numpy as np
from DataNexus.datahandler import DataHandler 
from DataNexus.fourier import Fourier
from HMM.hmm_classifier import HMMClassifier 
from sklearn.ensemble import RandomForestClassifier
import cPickle

# parameters
NSTATES = 2
NITERS = 2

# load the data
print "Reading data..."
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

# split the training data into two halves
#   fh stands for first half
#   sh stands for second half
print "Splitting data in two halves..."
fh_data, fh_labels, sh_data, sh_labels = DataHandler.split(0.5, train_data, train_labels)

# train HMM on first 50% of the training set
print "Training HMM classifier..."
hmmcl = HMMClassifier()
model_pos, model_neg = hmmcl.train(NSTATES, NITERS, fh_data, fh_labels)

# feed second 50% of the training set into the HMM to obtain
# pos/neg ratio for every sequence in the second half of the training set
print "Extracting ratios based on the HMM model..."
sh_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, sh_data)
test_ratios = hmmcl.pos_neg_ratios(model_pos, model_neg, test_data)

# apply fourier transform on the second 50% of the training set
print "Fouriering the second half of the dataset..."
fourier_sh_data = Fourier.data_to_fourier(sh_data)
fourier_test_data = Fourier.data_to_fourier(test_data)

# augment fourier results of the second 50% train with the ratios thus producing an enriched dataset
print "Merging Fourier features and HMM-based ratios..."
enriched_sh_data = np.hstack((fourier_sh_data, sh_ratios.reshape(len(sh_ratios), 1)))
enriched_test_data = np.hstack((fourier_test_data, test_ratios.reshape(len(test_ratios), 1)))

# train RF on the enriched dataset
print "Training RF on the merged dataset..."
rf = RandomForestClassifier(n_estimators=500)
rf.fit(enriched_sh_data, sh_labels)

# storing model into a file
print "Storing model into a file..."
with open("../../Results/models/rfhmm.pkl", "w") as f:
    cPickle.dump(rf, f)

# test RF on the test set
print "Testing the final model..."
print str(rf.score(enriched_sh_data, sh_labels)) + " - accuracy on train data"
print str(rf.score(enriched_test_data, test_labels)) + " - accuracy on test data"
