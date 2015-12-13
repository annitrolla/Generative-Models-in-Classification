import numpy as np
from LSTM.lstm_classifier import LSTMClassifier
                                   
def function(lstmsize, lstmdropout, lstmoptim):

    lstmsize = lstmsize[0] * 10
    lstmdropout = lstmdropout[0]
    lstmoptim = lstmoptim[0]
    lstmnepochs = 50
    lstmbatchsize = 64
    nfolds = 5

    print("Reading data...")
    dynamic_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_data.npy')
    labels_train = np.load('/storage/hpc_anna/GMiC/Data/ECoGmixed/preprocessed/train_labels.npy')
    nsamples = dynamic_train.shape[0]
   
    # k-fold cross validation to obtain accuracy
    val_idx_list = np.array_split(range(nsamples), nfolds)
    scores = []
    for fid, val_idx in enumerate(val_idx_list):
        train_idx = list(set(range(nsamples)) - set(val_idx))
        print "Current fold is %d / %d" % (fid + 1, nfolds)

        # LSTM on dynamic features (8)
        lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
        model_pos, model_neg = lstmcl.train(dynamic_train[train_idx], labels_train[train_idx])
        scores.append(lstmcl.test(model_pos, model_neg, dynamic_train[val_idx], labels_train[val_idx]))

    print 'Result: %.4f' % np.mean(scores)
    return -np.mean(scores)  
    
# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return function(params['lstmsize'], params['lstmdropout'], params['lstmoptim'])
