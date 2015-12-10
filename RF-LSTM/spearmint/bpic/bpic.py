
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from LSTM.lstm_classifier import LSTMClassifier
                                   
def bpic(lstmsize, lstmdropout, lstmoptim, nestimators):
    lstmsize = lstmsize[0] * 10
    lstmdropout = lstmdropout[0]
    lstmoptim = lstmoptim[0]
    lstmnepochs = 50
    lstmbatchsize = 256

    nestimators = nestimators[0] * 100
    nfolds = 5

    print lstmsize, lstmdropout, lstmoptim, nestimators

    # Load the dataset
    static_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_static.npy')
    dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_dynamic.npy')
    static_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_static.npy')
    dynamic_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_dynamic.npy')
    labels_train = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/train_labels.npy')
    labels_val = np.load('/storage/hpc_anna/GMiC/Data/BPIChallenge/f1/preprocessed/test_labels.npy')
    
    # Merge train and test
    static_all = np.concatenate((static_train, static_val), axis=0)
    dynamic_all = np.concatenate((dynamic_train, dynamic_val), axis=0)
    labels_all = np.concatenate((labels_train, labels_val), axis=0)
    nsamples = static_all.shape[0]

    # prepare where to store the ratios
    ratios_all_lstm = np.empty(len(labels_all))

    # split indices into folds
    enrich_idx_list = np.array_split(range(nsamples), nfolds)

    # run CV
    for fid, enrich_idx in enumerate(enrich_idx_list):
        train_idx = list(set(range(nsamples)) - set(enrich_idx))

        # extract predictions using LSTM on dynamic
        lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
        model_pos, model_neg = lstmcl.train(dynamic_all[train_idx], labels_all[train_idx])
        ratios_all_lstm[enrich_idx] = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all[enrich_idx])
    
    # dataset for hybrid learning
    enriched_by_lstm = np.concatenate((static_all, np.matrix(ratios_all_lstm).T), axis=1)
    
    # (2.) k-fold cross validation to obtain accuracy
    val_idx_list = np.array_split(range(nsamples), nfolds)
    scores = []
    for fid, val_idx in enumerate(val_idx_list):
        train_idx = list(set(range(nsamples)) - set(val_idx))
    
        # Hybrid on features enriched by HMM (3)
        rf = RandomForestClassifier(n_estimators=nestimators)
        rf.fit(enriched_by_lstm[train_idx], labels_all[train_idx])
        scores.append(rf.score(enriched_by_lstm[val_idx], labels_all[val_idx]))
    
    print 'Result: %.4f' % np.mean(scores)
    return -np.mean(scores)


# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return bpic(params['lstmsize'], params['lstmdropout'], params['lstmoptim'], params['nestimators'])
