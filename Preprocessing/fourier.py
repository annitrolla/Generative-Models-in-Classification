"""

Convert ECoG data to Fourier features

"""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import detrend
import matplotlib.pylab as plt


class Preprocessor:

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
        bins = [(start, start + binstep - 1) for start in range(0, ws, binstep)]

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

        plt.plot(features)
        plt.show()

        return np.array(features)


if __name__ == '__main__':
    data = np.loadtxt('../../Data/ECoG/Competition_train_cnt_sample.txt')
    for i in range(0, data.shape[0], 64):
        trial = data[i:i+64, 1:500].T
        sample = Preprocessor.monofourier(trial)
        print sample.shape
