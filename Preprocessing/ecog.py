"""
Handles ECoG dataset
"""

import numpy as np 
from scipy.io.netcdf import netcdf_file

class ECoG:

    trainval_data = None
    trainval_labels = None
    test_data = None
    test_labels = None

    def load_train_data(self):
        raw_data = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_cnt_scaled.txt")
        self.trainval_data = np.reshape(raw_data, (278, 64, 3000))
        self.trainval_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_train_lab_onezero.txt")

    def load_test_data(self):
        raw_data = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_cnt_scaled.txt")
        self.test_data = np.reshape(raw_data, (100, 64, 3000))
        self.test_labels = np.loadtxt("/storage/hpc_anna/GMiC/Data/ECoG/Competition_test_lab_onezero.txt")

    def split_train_val(self, ratio):
        idx_train = np.random.choice(range(0, self.trainval_data.shape[0]), size=int(self.trainval_data.shape[0] * ratio), replace=False)
        idx_val = list(set(range(0, self.trainval_data.shape[0])) - set(idx_train))
        train_data = self.trainval_data[idx_train, :, :]
        train_labels = self.trainval_labels[idx_train]
        val_data = self.trainval_data[idx_val, :, :]
        val_labels = self.trainval_labels[idx_val]    
        return train_data, train_labels, val_data, val_labels

    def dataset_to_netcdf(self, data, labels, filename):
        """
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
