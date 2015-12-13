import numpy as np
from HMM.hmm_classifier import HMMClassifier
                                   
def modelopt(nhmmstates, nhmmiter):

    nhmmstates = nhmmstates[0]
    nhmmiter = nhmmiter[0] * 10
    nfolds = 5
    hmmcovtype = 'full'

    print nhmmstates, nhmmiter

    # Load the dataset
    #static_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/train_data.npy')
    dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
    #static_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/fourier/test_data.npy')
    #dynamic_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_data.npy')
    labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
    #labels_test = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/test_labels.npy')
    nsamples = dynamic_train.shape[0]

    # k-fold cross validation to obtain accuracy
    val_idx_list = np.array_split(range(nsamples), nfolds)
    scores = []
    for fid, val_idx in enumerate(val_idx_list):
        train_idx = list(set(range(nsamples)) - set(val_idx))
        print "Current fold is %d / %d" % (fid + 1, nfolds)

        # extract predictions using HMM on dynamic
        hmmcl = HMMClassifier()
        model_pos, model_neg = hmmcl.train(nhmmstates, nhmmiter, hmmcovtype, 
                                           dynamic_train[train_idx], labels_train[train_idx])
        scores.append(hmmcl.test(model_pos, model_neg, dynamic_train[val_idx], labels_train[val_idx]))
    
    print 'Result: %.4f' % np.mean(scores)
    return -np.mean(scores)


# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return modelopt(params['nhmmstates'], params['nhmmiter'])
