# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy")
train_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_labels.npy")
test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy")
test_labels = np.load("/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_labels.npy")

rf = RandomForestClassifier(n_estimators=500, n_jobs=6)
rf.fit(train_data, train_labels)

print rf.score(train_data, train_labels)
print rf.score(test_data, test_labels)
