# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:20:41 2015

@author: annaleontjeva
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
from DataNexus.datahandler import DataHandler 

# load data
train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/fourier/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/fourier/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/fourier/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoG/fourier/test_labels.npy")

#pca = PCA(n_components=1000)
#pca.fit()
#pca.explained_variance_ratio_
#pca.explained_variance_
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#np.cumsum(pca.explained_variance_ratio_)[999]
#pca_data = pca.transform(data)
#pca_test_data = pca.transform(test_data)
# train = pca_data[0:4000,:]
# val = pca_data[4001:5560,:]

rf = RandomForestClassifier(n_estimators=500)
rf.fit(train_data, train_labels)

print rf.score(train_data, train_labels)
print rf.score(test_data, test_labels)
