import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier

nfolds = 5
nhmmstates = 10
nhmmiter = 70
nmix = 30
hmmcovtype = "full"  # options: full, diag, spherical

static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')


static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]

train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores_acc = []
scores_auc = []

for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(val_idx))
    # HMM on dynamic features (7)
    hmmcl = GMMHMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nmix, nhmmiter, hmmcovtype, dynamic_all[train_idx], labels_all[train_idx])
    acc, auc = hmmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx])
    scores_acc.append(acc)
    scores_auc.append(auc)

print "===> (7) accuracy of HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores_acc), np.std(scores_acc), scores_acc)
print "===> (7)      auc of HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores_auc), np.std(scores_auc), scores_auc)



