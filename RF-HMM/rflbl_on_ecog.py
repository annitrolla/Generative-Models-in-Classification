"""

The experiment combines the Fourier features and the features extraced from HMM generative model.
After this augmented dataset is created we train and evaluate Random Forest on it.

"""

import numpy as np
from DataNexus.datahandler import DataHandler 
from DataNexus.fourier import Fourier 
from sklearn.ensemble import RandomForestClassifier
import random

# load the data
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy")

#train_data = train_data[:30,:,:]
#train_labels = train_labels[:30]
#test_data = test_data[:30,:,:]
#test_labels = test_labels[:30]

# split the training data into two halves
#   fh stands for the first half
#   sh stands for the second half
#fh_data, fh_labels, sh_data, sh_labels = DataHandler.split(0.5, train_data, train_labels)
#np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/sh_labels.npy", sh_labels)
fourier_sh_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/fourier_sh_data.npy")
fourier_test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/fourier_test_data.npy")
sh_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/sh_labels.npy")

# introduce 42% of wrong answers to the labels
wrong_ratio = 0.10
wrong_idx = random.sample(range(len(sh_labels)), int(wrong_ratio * len(sh_labels)))
sh_labels_orig = list(sh_labels)
sh_labels[wrong_idx] = 0**sh_labels[wrong_idx]
print "Accuracy of the labels is %f" % (np.sum(sh_labels == sh_labels_orig) / float(len(sh_labels_orig)))

wrong_idx_test = random.sample(range(len(test_labels)), int(wrong_ratio * len(test_labels)))
test_labels_orig = list(test_labels)
test_labels[wrong_idx_test] = 0**test_labels[wrong_idx_test]

# apply fourier transform on the second 50% of the training set
#fourier_sh_data = Fourier.data_to_fourier(sh_data)
#fourier_test_data = Fourier.data_to_fourier(test_data)
#np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/fourier_sh_data.npy", fourier_sh_data)
#np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/fourier_test_data.npy", fourier_test_data)
#exit()

# augment fourier results of the second 50% train with the real labels thus producing an enriched dataset
enriched_sh_data = np.hstack((fourier_sh_data, sh_labels.reshape(len(sh_labels), 1)))
enriched_test_data = np.hstack((fourier_test_data, test_labels.reshape(len(test_labels), 1)))
#enriched_sh_data = fourier_sh_data
#enriched_test_data = fourier_test_data

# train RF on the enriched dataset
rf = RandomForestClassifier(n_estimators=500)
rf.fit(enriched_sh_data, sh_labels_orig)

# test RF on the test set
print str(rf.score(enriched_sh_data, sh_labels_orig)) + " - accuracy on train data"
print str(rf.score(enriched_test_data, test_labels_orig)) + " - accuracy on test data"
print "Importance of the features"
print list(rf.feature_importances_)[-5:]
