# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:40:04 2015

@author: annaleontjeva
"""

import numpy as np
from hmmlearn import hmm
import random as rm
from pylab import *
import matplotlib.pyplot as plt

# Example of data generation
startprob1 = np.array([0.9, 0.1]) # depends on number of states
transmat1 = np.array([[0.5, 0.5,], [0.5, 0.5]]) # nr states x nr states
means1 = np.array([[0.0, 0.0], [0.0, 0.0]]) # length of one box - number of features, boxes' number = nr states
covars1 = np.tile(np.identity(2), (2, 1, 1)) # identity nr = nr of features, first number in array = nr states
model1 = hmm.GaussianHMM(2, "full", startprob1, transmat1)
model1.means_ = means1
model1.covars_ = covars1
X1, Z1 = model1.sample(1000)
plot(X1)

startprob2 = np.array([0.5, 0.5])
transmat2 = np.array([[0.1, 0.9], [0.9, 0.1]])
means2 = np.array([[9.0, 8.0], [5.0, 6.0]])
covars2 = np.tile(np.identity(2), (2, 1, 1))
model2 = hmm.GaussianHMM(2, "full", startprob2, transmat2)
model2.means_ = means2
model2.covars_ = covars2
X2, Z2 = model2.sample(1000)
plot(X2)

# Mixing samples from two different models
labels = np.concatenate([np.repeat(0, len(X1)), np.repeat(1, len(X2))])
X3 = np.r_[X1, X2]
X3 = np.c_[X3, labels]
plot(X3)

rm.seed(a=3295)
idx = np.random.choice(range(0, len(X3)), size=np.round(len(X3)*1,0), replace=False)

test = X3[idx, :]
labels_test = test[:, X3.shape[1]-1]
data_test = test[:, 0:X3.shape[1]-1]
plot(data_test)


# Fitting HMM model
model3 = hmm.GaussianHMM(len(startprob1), covariance_type="full", n_iter=1000)
model3.fit([X1])

model4 = hmm.GaussianHMM(len(startprob2), covariance_type="full", n_iter=1000)
model4.fit([X2])


# Building Maximum likelihood classifier
def logdiff(x, model_true, model_false, data):
    if (model_true.score([data[x,:]]) >= model_false.score([data[x,:]])):
        return 0
    else:
        return 1

predictions = []
for i in range(data_test.shape[0]):
    predictions.append(logdiff(i, model3, model4, data_test))

predictions = np.array(predictions)

acc = float(sum(predictions == labels_test))/float(len(predictions))
print acc

k = data_test[labels_test==0,:] # 1 or 0
score_positive = []
score_negative = []

for i in range(len(k)):
    pos = model3.score([k[i,:]])
    neg = model4.score([k[i,:]])
    score_positive.append(pos) 
    score_negative.append(neg)

#results = np.c_[np.array([score_positive]), np.array([score_negative])]
plt.plot(score_positive)
plt.plot(score_negative)
plt.show()

# Automatically find number of hidden states
# ..split train data on two pices

# data=X3

def class_division(data):
    labels = data[:,data.shape[1]-1]
    pos = data[labels==1, :]
    pos = np.delete(pos,-1,1)
    neg = data[labels==0, :]
    neg = np.delete(neg,-1,1)
    return pos, neg
    
def random_split(data=X3, ratio=0.5, seed=9535):
    rm.seed(a=seed)
    idx_train = np.random.choice(range(0, len(data)), size=np.round(len(data)*ratio,0), replace=False)
    idx_val = list(set(range(0, len(data)))- set(idx_train))
    train_set = data[idx_train, :]
    val_set = data[idx_val, :]
    train_pos, train_neg = class_division(train_set)
    val_pos, val_neg = class_division(val_set)
    return train_pos, train_neg, val_pos, val_neg

train_pos, train_neg, val_pos, val_neg = random_split(ratio=0.5)
labels_val = np.concatenate([np.repeat(1, len(val_pos)), np.repeat(0, len(val_neg))])
val = np.r_[val_pos, val_neg]

# Building Maximum likelihood classifier
def logdiff(x, model_pos, model_neg, data):
    if (model_pos.score([data[x,:]]) >= model_neg.score([data[x,:]])):
        return 1
    else:
        return 0

def predictions(data_test, model_pos, model_neg, labels_test, show_prediction=False):
    pred = []
    for i in range(len(data_test)):
        pred.append(logdiff(i, model_pos, model_neg, data_test))
    pred = np.array(pred)
    acc = float(sum(pred == labels_test))/float(len(pred))
    if show_prediction == True:
        return acc, pred
    else:
        return acc
    
# building models
hidden_state_range = range(2,10)
accuracy_results = []
for state in hidden_state_range:
    model_pos = hmm.GaussianHMM(state, covariance_type="full", n_iter=1000)
    model_pos.fit([train_pos])
    model_neg = hmm.GaussianHMM(state, covariance_type="full", n_iter=1000)
    model_neg.fit([train_neg])
    # validation
    accuracy = predictions(val, model_pos=model_pos, model_neg=model_neg, labels_test=labels_val)
    accuracy_results.append(accuracy)
    
best_state = hidden_state_range[max(enumerate(accuracy_results), key=lambda x: x[1])[0]]


# Now use real test data

model_pos = hmm.GaussianHMM(best_state, covariance_type="full", n_iter=1000)
model_pos.fit([train_pos])
model_neg = hmm.GaussianHMM(best_state, covariance_type="full", n_iter=1000)
model_neg.fit([train_neg])
# validation
accuracy, predictions = predictions(np.delete(test,-1,1), model_pos=model_pos, model_neg=model_neg, labels_test=labels_test, show_prediction=True)




# Check if fitting the data changes the model. It does

model3.score([data_test[2,:]])  
model4.score([data_test[2,:]]) # Log probability of the maximum likelihood path through the HMM
labels_test[2]

model3.transmat_ #transition matrix
model4.transmat_

model3.startprob_
model4.startprob_

model3.means_ # Mean parameters for each state.
model4.means_

hidden_states1 = model3.predict(X1)
hidden_states2 = model4.predict(X2)


# Auxilary information

# model2.score_samples(X)[1].shape
#Z2 = model2.predict(X)
#model3.score_samples(X) # Compute the log probability under the model and posteriors.