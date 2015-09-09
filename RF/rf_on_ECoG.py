# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:20:41 2015

@author: annaleontjeva
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
from DataNexus.datahandler import DataHandler 

dh = DataHandler('/storage/hpc_anna/GMiC/Data/ECoG/fourier')
dh.load_train_data()
dh.load_test_data()

   
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
rf.fit(dh.trainval_data, dh.trainval_labels)

print rf.score(dh.trainval_data, dh.trainval_labels)
print rf.score(dh.test_data, dh.test_labels)
