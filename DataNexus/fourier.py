"""

Convert ECoG data to Fourier features

"""

import sys
import numpy as np
from scipy.fftpack import fft
from scipy.signal import detrend
from DataNexus.datahandler import DataHandler


class Fourier:

    def __init__(self):
        pass

    @staticmethod
    def monofourier(rawdata):
        """ Apply FFT on the whole sample and return the resulting instance (one)"""

        # signal parameters
        sfreq = 1000.0
        ws = rawdata.shape[0]
        binstep = int(sfreq / ws)
        fftcutoff = 150

        # one bin should correspond to 1Hz regardless of windows size and sampling frequency
        bins = [(start, start + binstep - 1) for start in range(0, fftcutoff, binstep)]

        # transform data
        channels = rawdata.shape[1]

        # initialize features vector
        features = np.array([])

        # go over channels
        for ch in range(0, channels):

            # detrend and demean the signal
            signal = detrend(rawdata[:, ch])

            # transform to frequency domain
            freqspace = abs(fft(signal))

            # extracts bands specified in the description of the device
            representation = []
            for bin in bins:
                representation += [np.mean(freqspace[bin[0]:bin[1] + 1])]
            representation = representation[0:fftcutoff]

            # append to current feature vector
            features = np.hstack((features, representation))

        #plt.plot(features)
        #plt.show()

        return np.array(features)

    @staticmethod
    def data_to_fourier(data):
        print "Transforming a dataset to Fourier space..."
        
        fourier_data = None
        nsamples = data.shape[0]
        
        for i, sample in enumerate(data):
            
            # display progress
            sys.stdout.write('{0}/{1}\r'.format(i, nsamples))
            sys.stdout.flush()

            # transform the sample in Fourier space
            instance = Fourier.monofourier(sample.T)

            # lazily initialize results matrix
            if fourier_data is None:
                fourier_data = np.empty((data.shape[0], len(instance)))

            # store the transformed sample to the resulting dataset
            fourier_data[i] = instance

        return fourier_data


if __name__ == '__main__':

    print "Loading training data..."
    train_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy")
    print "Loading test data..."
    test_data = np.load("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy")

    fourier_data = Fourier.data_to_fourier(train_data)
    np.save('/storage/hpc_anna/GMiC/Data/ECoG/fourier/train_data.npy', fourier_data)

    fourier_data = Fourier.data_to_fourier(test_data)
    np.save('/storage/hpc_anna/GMiC/Data/ECoG/fourier/test_data.npy', fourier_data)
    

            
            
