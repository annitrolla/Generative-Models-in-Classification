"""
Handles ECoG dataset
"""

import numpy as np
from DataNexus.datahandler import DataHandler 
from scipy.io.netcdf import netcdf_file


class ECoGPreprocessor:

    def dataset_to_netcdf(self, data, labels, filename):
        """
        BROKEN
        Given a dataset in numpy tensor format store it as a NetCDF structure
        """

        # extract dimensions
        nsubjects = data.shape[0]
        nfeatures = data.shape[1]
        ntime = data.shape[2]

        # reshape from (SUBJECTS, FEATURES, TIME)
        #           to (SUBJECTS x TIME, FEATURES)
        reshaped = data.swapaxes(1,2).reshape(nsubjects * ntime, nfeatures)
        
        # build .nc file for the training set
        nc = netcdf_file('/storage/hpc_anna/GMiC/Data/ECoG/%s' % filename, 'w')
        nc.createDimension('numSeqs', nsubjects)
        nc.createDimension('numTimesteps', nsubjects * ntime)
        nc.createDimension('inputPattSize', nfeatures)
        nc.createDimension('numLabels', 2)
        nc.createDimension('maxSeqTagLength', 100)
        nc.createDimension('one', 1)

        nc_inputs = nc.createVariable('inputs', np.dtype('float32').char, ('numTimesteps', 'inputPattSize'))
        nc_inputs[:] = reshaped

        nc_numTargetClasses = nc.createVariable('numTargetClasses', np.dtype('int32').char, ('one', ))
        nc_numTargetClasses[:] = 2

        nc_seqLengths = nc.createVariable('seqLengths', np.dtype('int32').char, ('numSeqs', ))
        nc_seqLengths[:] = [ntime] * nsubjects

        nc_targetClasses = nc.createVariable('targetClasses', np.dtype('int32').char, ('numTimesteps', ))
        nc_targetClasses[:] = np.array([[int(x)] * ntime for x in labels]).reshape(nsubjects * ntime)

        nc_seqTags = nc.createVariable('seqTags', '|S1', ('numSeqs', 'maxSeqTagLength'))

        nc.close()

    def save_netcdf(self):
        """
        BROKEN
        Convert the ECoG data to NetCDF files
        """

        # read in data files
        self.load_train_data()
        self.load_test_data()

        # split trainval to train and val
        train_data, train_labels, val_data, val_labels = self.split_train_val(ratio=0.7)
	
        # store training, validation and test as NetCDF
        self.dataset_to_netcdf(train_data, train_labels, 'train.nc')
        self.dataset_to_netcdf(val_data, val_labels, 'val.nc')
        self.dataset_to_netcdf(self.test_data, self.test_labels, 'test.nc')

    @staticmethod
    def scale(data):
        for i in range(data.shape[0]):
            for c in range(data.shape[1]):
                data[i, c, :] = (data[i, c, :] - np.mean(data[i, c, :])) / np.std(data[i, c, :])
        return data

    @staticmethod
    def slice(data, labels, winsize, winstep):
        sliced = np.empty((data.shape[0] * ((data.shape[2] - winsize) / winstep + 1), data.shape[1], winsize))
        newlabels = np.empty((data.shape[0] * ((data.shape[2] - winsize) / winstep + 1)))
        counter = 0
        for i in range(data.shape[0]):
            for s in range(0, data.shape[2] - winsize + 1, winstep):
                sliced[counter, :, :] = data[i, :, s:s+winsize].reshape(1, data.shape[1], winsize)
                newlabels[counter] = labels[i]
                counter += 1
        return sliced, newlabels


if __name__ == '__main__':
    """
    Default preprocessing pipeline for the ECoG dataset. Preprocessed files are stored to predefined location.
    If you need to try out another preprocessing pipeline you'll to change this function.
    """
    
    print "Loading raw training data..."
    train_raw = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/raw/train_data.txt")
    train_raw = np.reshape(train_raw, (278, 64, 3000))
    train_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/raw/train_labels.txt")

    print "Slicing training data..."
    train_sliced, train_labels = ECoGPreprocessor.slice(train_raw, train_labels, 300, 100)

    print "Scaling training data..."
    train_scaled = ECoGPreprocessor.scale(train_sliced)

    print "Storing training dataset..."
    np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_data.npy", train_scaled)
    np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/train_labels.npy", train_labels)

    print "Loading raw test data..."
    test_raw = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/raw/test_data.txt")
    test_raw = np.reshape(test_raw, (100, 64, 3000))
    test_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/raw/test_labels.txt")

    print "Slicing test data..."
    test_sliced, test_labels = ECoGPreprocessor.slice(test_raw, test_labels, 300, 100)

    print "Scaling test data..."
    test_scaled = ECoGPreprocessor.scale(test_sliced)

    print "Storing test dataset..."
    np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_data.npy", test_scaled)
    np.save("/storage/hpc_anna/GMiC/Data/ECoG/preprocessed/test_labels.npy", test_labels)

    print "Done."











