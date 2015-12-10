import numpy as np
from HMM.hmm_classifier import HMMClassifier

# parameters
nfolds = 5
nstates = 6
niter = 50

# load data
static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')
nsamples = dynamic_train.shape[0]

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d/%d" % (fid + 1, nfolds)
    train_idx = list(set(range(nsamples)) - set(val_idx))

    # train the model and report performance
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nstates, niter, 'full', dynamic_train[train_idx], labels_train[train_idx])
    scores.append(hmmcl.test(model_pos, model_neg, dynamic_train[val_idx], labels_train[val_idx]))

print "===> (7) HMM with dynamic features on CV: %.4f (+/- %.4f) %s" % (np.mean(scores), np.std(scores), scores)
