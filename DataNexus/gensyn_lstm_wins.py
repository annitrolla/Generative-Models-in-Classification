"""

Generate a dataset with the following properties
    - Binary classification
    - Static features which are ~0.7 separable using RF
    - Dynamic features which are ~0.7 separable using generative-model-based classifier
    - Some of the instances are predicatable only from the static features, other only from
      the sequence data, other from both, and some are not reasonably predictable at all
    - All features are generated from the SAME distribution

"""

import numpy as np
import statsmodels.tsa.arima_process as ap

def generate_lstm_wins(nsamples, nfeatures, nseqfeatures, seqlen):
    
    # size of 1/4th of a dataset
    nq = nsamples / 4
    
    # dynamical feature distribution parameters (for arma)
    orders = [np.random.randint(1, 5, 4) for rep in range(nseqfeatures)]
    arparams_pos = [np.hstack((1.0, np.random.uniform(-0.1, 0.1, ord[0]))) for ord in orders]
    arparams_neg = [np.hstack((1.0, np.random.uniform(-0.1, 0.1, ord[1]))) for ord in orders]
    maparams_pos = [np.hstack((1.0, np.random.uniform(-0.5, 0.5, ord[2]))) for ord in orders]
    maparams_neg = [np.hstack((1.0, np.random.uniform(-0.5, 0.5, ord[3]))) for ord in orders]

    #
    # Generate data
    #
    print 'Generating data...'

    # generate labels
    labels = np.array([1] * nq + [0] * nq + [1] * nq + [0] * nq)

    # generate distribution parameters for the static features
    smean_pos = np.random.uniform(0, 2, nfeatures)
    ssigm_pos = [1.0] * nfeatures
    smean_neg = np.random.uniform(0, 2, nfeatures)
    ssigm_neg = [1.0] * nfeatures

    # generate static features
    static_q1 = np.vstack([np.random.normal(smean_pos[f], ssigm_pos[f], nq) for f in range(nfeatures)])
    static_q2 = np.vstack([np.random.normal(smean_neg[f], ssigm_neg[f], nq) for f in range(nfeatures)])
    static_q3 = np.vstack([np.random.normal(smean_neg[f], ssigm_neg[f], nq) for f in range(nfeatures)])
    static_q4 = np.vstack([np.random.normal(smean_neg[f], ssigm_neg[f], nq) for f in range(nfeatures)])
    static = np.vstack((static_q1.T, static_q2.T, static_q3.T, static_q4.T))

    # generate dynamic features
    dynamic_q1 = np.vstack([ap.arma_generate_sample(arparams_pos[i % nseqfeatures], maparams_pos[i % nseqfeatures], seqlen)
                           for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen) 
    dynamic_q2 = np.vstack([ap.arma_generate_sample(arparams_pos[i % nseqfeatures], maparams_pos[i % nseqfeatures], seqlen)
                           for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
    dynamic_q3 = np.vstack([ap.arma_generate_sample(arparams_neg[i % nseqfeatures], maparams_neg[i % nseqfeatures], seqlen)
                           for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
    dynamic_q4 = np.vstack([ap.arma_generate_sample(arparams_pos[i % nseqfeatures], maparams_pos[i % nseqfeatures], seqlen)
                           for i in range(nq * nseqfeatures)]).reshape(nq, nseqfeatures, seqlen)
    dynamic = np.vstack((dynamic_q1, dynamic_q2, dynamic_q3, dynamic_q4))


    #
    # Split the dataset into training and test sets
    #
    print 'Splitting train and test...'

    # pick samples for training and for testing
    train_idx = np.random.choice(range(0, nsamples), size=np.round(nsamples * 0.7, 0), replace=False)
    test_idx = list(set(range(0, nsamples)) - set(train_idx))

    train_static = static[train_idx, :]
    train_dynamic = dynamic[train_idx, :, :]
    test_static = static[test_idx, :]
    test_dynamic = dynamic[test_idx, :, :]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    return train_static, train_dynamic, test_static, test_dynamic, train_labels, test_labels
   

if __name__ == "__main__":
    train_static, train_dynamic, test_static, test_dynamic, train_labels, test_labels = generate_lstm_wins(1000, 50, 100, 30)
    print 'Storing the dataset...'
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_static.npy', train_static)
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_dynamic.npy', train_dynamic)
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_static.npy', test_static)
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_dynamic.npy', test_dynamic)
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/train_labels.npy', train_labels)
    np.save('/storage/hpc_anna/GMiC/Data/syn_lstm_wins/test_labels.npy', test_labels)
    print 'Done.'

