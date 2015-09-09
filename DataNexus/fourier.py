"""

Convert ECoG data to Fourier features

"""

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


if __name__ == '__main__':
    dh = DataHandler("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed")
    dh.load_train_data()
    dh.load_test_data()
    
    # transforming train data to fourier
    fourier_data = np.empty((dh.train_data.shape[0], 3200))
    for i, sample in enumerate(dh.train_data): 
        fourier_data[i] = Fourier.monofourier(sample.T)
    np.save('/storage/hpc_anna/GMiC/Data/ECoG/fourier/train_data.npy', fourier_data)

    # transforming test data to fourier
    fourier_data = np.empty((dh.test_data.shape[0], 3200))
    for i, sample in enumerate(dh.test_data):
        fourier_data[i] = Fourier.monofourier(sample.T)
    np.save('/storage/hpc_anna/GMiC/Data/ECoG/fourier/test_data.npy', fourier_data)
    

            
            
