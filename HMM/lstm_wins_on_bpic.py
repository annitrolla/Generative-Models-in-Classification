import numpy as np
from sklearn.ensemble import RandomForestClassifier
from HMM.hmm_classifier import HMMClassifier, GMMHMMClassifier

nfolds = 5
nhmmstates = 10
nhmmiter = 200
hmmcovtype = "full"  # options: full, diag, spherical

static_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy')
dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy')
static_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy')
dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy')
labels_train = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy')
labels_val = np.load('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy')


static_all = np.concatenate((static_train, static_val), axis=0)
dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
labels_all = np.concatenate((labels_train, labels_val), axis=0)
nsamples = static_all.shape[0]

train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
test_idx = list(set(range(0, nsamples)) - set(train_idx))

# split indices into folds
val_idx_list = np.array_split(range(nsamples), nfolds)

# run CV
scores = []
for fid, val_idx in enumerate(val_idx_list):
    print "Current fold is %d" % fid
    train_idx = list(set(range(nsamples)) - set(val_idx))
    # HMM on dynamic features (7)
    hmmcl = HMMClassifier()
    model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, dynamic_all[train_idx], labels_all[train_idx])
    scores.append(hmmcl.test(model_pos, model_neg, dynamic_all[val_idx], labels_all[val_idx]))

print "===> (7) HMM on dynamic features: %.4f (+/- %.4f) %s" % (np.mean(scores), np.std(scores), scores)




