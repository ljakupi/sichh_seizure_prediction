import re
import os
import argparse
import numpy as np

'''
  The config class defines the the parameters for the handling of data and training. This input file 
  was created for patient 1 of the CHB-MIT, and should probably be adapted for other patients datasets.
  For instance, the number of channels could change. A text file could store the information for specific
  dataset and be used to define the relevant files and channels.

  data_path: path to the folder with features files
  self.patient: number of the patient (in CHB-MIT the patient are numbered)
  self.features: which features to use, corresponding to file in data_path
  self.NN: the type of neural network
  self.segments_type_train: type of segments for training (random, continuous, pattern)
  self.segments_type_test: type of segments for testing (random, continuous, pattern)
  self.sampling: sampling rate of the inputs files (in Hz)
  self.duration: length of segments of the input files
  self.test_len: number of sequences used for testing
  self.preictal_duration: time in seconds at which we define segments to be in the preictal period
  self.learning_rate: learning rate of the optimizer
  self.training_steps: number of steps used for the training of the neural network
  self.batch_size: number of sequences used for training
  self.display_step: display loss and accuracy every n steps
  self.num_inputs: number of segments put together to make a sequence
  self.num_hidden: number of neurons in the hidden layers (for LSTM and FC neural networks)
  self.levels: depth of the neural network (TCN only)
  self.num_classes: number of classes (usually preictal and interictal)
  self.channels_names: names of channels used (it should correspond to the ones used to create the features)
  self.selected_channels: choose which channel will be used. Only the selected channels will be loaded
  self.m_features: number of 'rows' for each feature. It is usually 1 but if the features has more than
                    one frequency, this number equals the number of frequencies
  self.u_features: number of univartiate features
  self.N_features: number of features, this number is equal to the sum of m_features times N_channels*(N_channels-1)/2 + self.u_features*self.N_channels
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to load feature files (.dat)")
    parser.add_argument("--patient", help="Patient number")
    parser.add_argument("--preictal_duration",
                        help="Time in seconds at which we define segments to be in the preictal period (e.g. 3600 sec = 30 min)")
    parser.add_argument("--discard_data", help="Choose if we discard the data or not, value bool: True or False")
    parser.add_argument("--discard_data_duration",
                        help="Criteria to discard some data before and after seizure, expressed in minutes (e.g. 240 min = 4 hours) ")
    parser.add_argument('--features_names', nargs='+',
                        help="Select the features we want to include for dataset preparation")  # This is the correct way to handle accepting multiple arguments => '+' == 1 or more
    args = parser.parse_args()

    DataPreparation(args)


class DataPreparation():
    '''
      Class Patient_data loads data in the following format:
            seizure_1_start seizure_1_end seizure_2_start ...
            time_1 feature_1 feature_2 feature_3 ...
            time_2 feature_1 feature_2 feature_3 ...
      seizures_times are stored in $seizures_start and $seizures_end
      features are stored in $segments (without time)
      times are stored separately to feature in $times array
    '''

    def __init__(self, args):

        self.data_path = args.data_path
        self.patient = int(args.patient)

        # load feature files
        self.features_files = []
        for feature in args.features_names:
            self.features_files.append(args.data_path + '/' + 'chb{:02d}'.format(self.patient) + '/' + feature + '.dat')

        # channels used from raw EDF data to calculate features
        self.channels_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4",
                               "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2", "FZ-CZ", "CZ-PZ"]
        self.selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

        # set feature types
        self.feature_type = {}
        self.feature_type["max_correlation"] = "bivariate"
        self.feature_type["DSTL"] = "bivariate"
        self.feature_type["nonlinear_interdependence"] = "bivariate"
        self.feature_type["SPLV"] = "bivariate"
        self.feature_type["univariate"] = "univariate"

        # set multivariate features number
        self.m_features = 0
        self.m_feature = {}
        self.m_feature["max_correlation"] = 1
        self.m_feature["SPLV"] = 7
        self.m_feature["nonlinear_interdependence"] = 1
        self.m_feature["DSTL"] = 1

        # set univariate features number
        self.u_features = 0
        self.u_feature = {}
        self.u_feature["univariate"] = 24

        # prepare multivariate and univariate features
        for feature in args.features_names:
            if feature in self.m_feature:
                self.m_features += self.m_feature[feature]
            if feature in self.u_feature:
                self.u_features += self.u_feature[feature]

        self.preictal_duration = int(args.preictal_duration)
        self.discard_data = bool(args.discard_data)
        self.discard_data_duration = int(args.discard_data_duration)

        # calculate the number of features
        self.N_channels = len(self.selected_channels)
        self.N_features = int(self.N_channels * (self.N_channels - 1) / 2.) * self.m_features
        self.N_features += int(
            self.N_channels * self.u_features)  # if univariate features involved than N_features changes

        # At this step, variables:
        # self.segments:
        # - Shape: n_segments x n_features, e.g. 25353 x 306 where 25353 are all the rows/segments_5_sec/data-points in .dat files and 306 is the number of all feature files concatenated e.g. if two feature files with 153 columns in each file, than we have in total 306 features.
        # - Content: It holds all the segments/data-points with their feature columns excluding the first column which is the time
        # self.times: array that holds all the times (first column in feature file (.dat file)) - all the .dat files have the same times
        # self.seizures_start: holds patient's start times of every seizure
        # self.seizures_end: holds patient's end times of every seizure
        self.segments, self.times, self.seizures_start, self.seizures_end = self.loadAllFiles(self.features_files)

        self.N_seizures = len(self.seizures_start)
        print("Patient ", args.patient, " has ", len(self.seizures_start), ' seizures')

        # Array of values 0 (interictal), 1 (preictal), 2 (ictal) for every time/row (segments.shape[0])
        self.labels = self.labelData()

    def loadAllFiles(self, input_files):
        '''
          load all_files (one per feature). The feature name is taken from the
          input file name. Only the feature corresponding to selected channels are stored.
        '''
        end = 0
        for i in range(0, len(input_files)):  # E.g. input_files: ['./chb02/max_correlation.dat', './chb02/DSTL.dat']
            feature = re.search('chb[0-9]*/(.*).dat', input_files[i]).group(
                1)  # feature file name (.dat file name) - e.g. max_correlation, DSTL, SPLV
            idx = self.get_indices(self.selected_channels, feature).astype(
                int)  # features/columns indices (columns in .dat file) - e.g. max_correlation has 153 indices, SPLV has 1071 indices

            # CHECK AND FIX THIS CONDITION! LOOKS WEIRD.
            if (i == 0):
                segments_i, times, seizures_start, seizures_end = self.loadFileData(input_files[i])
                shape = np.shape(segments_i)
                segments = np.empty([shape[0], self.N_features])
            else:
                segments_i, _, _, _ = self.loadFileData(input_files[i])

            segments_i = segments_i[:, idx]
            shape = np.shape(segments_i)
            start = end
            end += shape[1]

            segments[:,
            start:end] = segments_i  # concatenate all features per segment. E.g. if we use 2 feature files and each file have 153 feature columns, than the concatenation will result in 306 column features

            print("Loaded feature file: ", feature)
        print("Total segments shape (after concatenation of all feature files): ", np.shape(segments))
        # ~ start_uni = self.N_features - 18 * 25
        # ~ self.N_features -= 18 * 20
        # ~ print(segments[:,start_uni+18*5:start_uni+18*6])
        # ~ segments = np.delete(segments, np.s_[start_uni+18*5:start_uni+18*6],axis=1)
        self.N_features = np.shape(segments)[1]
        # ~ segments=segments[:,:-18*20]
        return segments, times, seizures_start, seizures_end

    def get_indices(self, selected_channels, feature):
        '''
        input: selected channels, and feature name
        output: indices corresponding to upper diagonal for selected channels in config
        '''
        feature_type = self.feature_type[feature]
        N_tot_channels = len(self.channels_names)
        selected_channels_bool = np.array([False] * N_tot_channels)
        selected_channels_bool[selected_channels] = True
        idx = np.empty(0, dtype=int)
        idx_one = np.empty(0, dtype=int)
        if feature_type == "bivariate":
            m = self.m_feature[feature]
            c = 0
            for i in range(0, N_tot_channels - 1):
                for j in range(i + 1, N_tot_channels):
                    if (selected_channels_bool[i] and selected_channels_bool[j]):
                        idx_one = np.append(idx_one, c)
                    c += 1
            N_one_feature = int(N_tot_channels * (N_tot_channels - 1) / 2.)
            for i in range(0, m):
                idx = np.append(idx, idx_one + N_one_feature * i)

        elif feature_type == "univariate":
            u = self.u_feature[feature]
            idx_one = np.array(selected_channels)
            for i in range(0, u):
                idx = np.append(idx, idx_one + N_tot_channels * i)

        return idx

    def labelData(self):
        # 0 - interictal
        # 1 - preictal
        # 2 - ictal
        labels = np.zeros(len(self.times), dtype=int)

        discard_data = self.discard_data # True or False
        discard_data_duration = self.discard_data_duration  # segments within this time duration are discarded. E.g. 240min * 60 = 4 hours

        # lists to hold labeled segments
        preictal_list = []
        ictal_list = []
        interictal_list = []

        for index, seizure_start in enumerate(self.seizures_start):
            for i in range(0, len(self.times)):
                if (discard_data):
                    if (seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
                        labels[i] = 1  # preictal
                        preictal_list.append(self.segments[i])


                    elif (self.seizures_start[index] < self.times[i] and self.seizures_end[index] > self.times[i]):
                        labels[i] = 2  # ictal
                        ictal_list.append(self.segments[i])


                    elif (seizure_start > self.times[i] and (
                                    (seizure_start - self.preictal_duration) - self.times[
                                    i] < discard_data_duration) or (
                                (self.seizures_end[index] + discard_data_duration) > self.times[i])):  # discard_data_duration
                        interictal_list.append(self.segments[i])

                else:
                    if (seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
                        labels[i] = 1  # preictal
                    elif (self.seizures_start[index] < self.times[i] and self.seizures_end[index] > self.times[i]):
                        labels[i] = 2  # ictal

        np_preictal_file = np.asarray(preictal_list, dtype=np.float32)
        np_ictal_file = np.asarray(ictal_list, dtype=np.float32)
        np_interictal_file = np.asarray(interictal_list, dtype=np.float32)

        # writing labeled segments to separate binary files
        parent_dir = './processed_datasets/CHB-MIT/' + 'chb{:02d}'.format(self.patient) + '/preictal-' + str(int(self.preictal_duration / 60))

        # if binary fiels' parent dir does not exist, then create
        os.makedirs(parent_dir, exist_ok=True)

        # set files' names
        preictal_file = parent_dir + '/preictal_segments.npy'
        ictal_file = parent_dir + '/ictal_segments.npy'
        interictal_file = parent_dir + '/interictal_segments.npy'

        # save numpy arrays to binary files
        np.save(preictal_file, np_preictal_file)
        np.save(ictal_file, np_ictal_file)
        np.save(interictal_file, np_interictal_file)

        # check how many data-points are annotated as 1 (preictal) or 2 (ictal) and 0 (interictal) respectively
        unique, counts = np.unique(labels, return_counts=True)
        print('Total samples grouped by class (labels: interictal=0, preictal=1, ictal=2): ',
              dict(zip(unique, counts)))

        return labels  # array filled with values 0, 1, 2

    def loadFileData(self, input_file):
        f = open(input_file, 'r')
        line = f.readline()
        seizures_start = []
        seizures_end = []
        seizures_time = [int(t) for t in line.strip().split(
            ' ')];  # get the seizure times - first row/array in .dat files. This row is a pair of values indicating the start and end time of a seizure
        for i in range(0, int(len(
                seizures_time) / 2)):  # /2 because if there are 3 seizures there are 6 values in seizure times (seizures_time) with start and end time for each seizure
            seizures_start.append(seizures_time[
                                      2 * i])  # append each seizure start time to seizures_start array. E.g if there are 6 times, then 0, 2, 4 are the starting times
            seizures_end.append(seizures_time[
                                    2 * i + 1])  # append each seizure end time to seizures_end array. E.g if there are 6 times, then 1, 3, 5 are the ending times

        lines = [line.rstrip().split(' ') for line in f]
        # N_features = len(lines[0]) - 1 # weird variable - not used and still is declared

        # segments = np.empty([len(lines), N_features + 1]) # commented because not used ---> initialize an empty-valued array that will hold all the segments (one segment = 5 sec long).
        segments = np.float32(lines)  # assign segments with values from feature files (.dat)

        times = segments[:, 0]  # store times (first column in .dat file)
        segments = segments[:,
                   1:]  # store segments (all the columns excluding the first one (time column)) - one segment have many columns
        return segments, times, seizures_start, seizures_end


if __name__ == '__main__':
    main()
