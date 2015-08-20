# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:20:41 2015

@author: annaleontjeva
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA 
import cPickle
import numpy as np

with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_fourier_labels.pkl', 'rb') as f:
    labels = cPickle.load(f) 
    
with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_fourier_data.pkl', 'rb') as f:
    data = cPickle.load(f)

with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_fourier_labels.pkl', 'rb') as f:
    test_labels = cPickle.load(f) 
    
with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_fourier_data.pkl', 'rb') as f:
    test_data = cPickle.load(f)


data = np.array(data)    
test_data = np.array(test_data) 
   
pca = PCA(n_components=1000)
pca.fit(data)
#pca.explained_variance_ratio_
#pca.explained_variance_
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
np.cumsum(pca.explained_variance_ratio_)[999]

pca_data = pca.transform(data)
pca_test_data = pca.transform(test_data)

#with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_pca_data.pkl', 'wb') as f:
#    cPickle.dump(pca_data, f)
    
#with open('/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_pca_data.pkl', 'wb') as f:
#    cPickle.dump(pca_test_data, f)

# train = pca_data[0:4000,:]
# val = pca_data[4001:5560,:]

rf = RandomForestClassifier(n_estimators=10)
rf.fit(pca_data, labels)

print rf.score(pca_data, labels)
print rf.score(pca_test_data, test_labels)