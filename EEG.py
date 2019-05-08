from __future__ import print_function
import tensorflow as tf
#from tensorflow.contrib import rnn
import multiprocessing as mp
import ctypes
import datetime
import time
import bisect
import numpy as np
import scipy
import scipy.stats
import random
import re
# import matplotlib.pyplot as plt
# import pywt
# import networkx as nx


######## additional imports
# from sklearn.utils import shuffle
from itertools import zip_longest, islice, chain
########


random.seed(1)


import config

def main():
    cfg = config.Config(data_path='./data_features/CHB-MIT_features', NN='FC', patient=int(2))
    Patient_data(cfg)


class Patient_data():
    '''
      Class Patient_data loads data in the following format:
            seizure_1_start seizure_1_end seizure_2_start ...
            time_1 feature_1 feature_2 feature_3 ...
            time_2 feature_1 feature_2 feature_3 ...
      seizures_times are stored in $seizures_start and $seizures_end
      features are stored in $segments (without time)
      times are stored separately to feature in $times array
    '''

    def __init__(self, cfg):
        '''
          Here the parameters important for the class are copied, so it is clear which paramters are relevant here
          (However it may be simpler to change it to self.cfg=cfg)
        '''
        self.data_path = cfg.data_path
        self.segments_type_train = cfg.segments_type_train
        self.segments_type_test = cfg.segments_type_test
        self.preictal_duration = cfg.preictal_duration
        self.channels_names = cfg.channels_names
        self.selected_channels = cfg.selected_channels
        self.N_tot_features = cfg.N_tot_features
        self.num_inputs = cfg.num_inputs
        self.feature_type = cfg.feature_type
        self.N_features = cfg.N_features
        self.m_feature = cfg.m_feature
        self.u_feature = cfg.u_feature
        self.num_classes = cfg.num_classes

        # At this step, variables:
        # self.segments:
            # - Shape: n_segments x n_features, e.g. 25353 x 306 where 25353 are all the rows/segments_5_sec/data-points in .dat files and 306 is the number of all feature files concatenated e.g. if two feature files with 153 columns in each file, than we have in total 306 features.
            # - Content: It holds all the segments/data-points with their feature columns excluding the first column which is the time
        # self.times: array that holds all the times (first column in feature file (.dat file)) - all the .dat files have the same times
        # self.seizures_start: holds patient's start times of every seizure
        # self.seizures_end: holds patient's end times of every seizure
        self.segments, self.times, self.seizures_start, self.seizures_end = self.load_all_files(cfg.input_files)

        self.N_seizures = len(self.seizures_start)
        print("Patient ", cfg.patient, " has ", len(self.seizures_start), ' seizures')

        # Array of values 0 (interictal), 1 (preictal), 2 (ictal) for every time/row (segments.shape[0])
        self.annotations = self.annotate_data()

        self.leave_one_seizure(1)
        # ~ self.k_fold_division(4)

    def load_all_files(self, input_files):
        '''
          load all_files (one per feature). The feature name is taken from the
          input file name. Only the feature corresponding to selected channels are stored.
        '''
        end = 0
        for i in range(0, len(input_files)): # E.g. input_files: ['./chb02/max_correlation.dat', './chb02/DSTL.dat']
            feature = re.search('chb[0-9]*/(.*).dat', input_files[i]).group(1) # feature file name (.dat file name) - e.g. max_correlation, DSTL, SPLV
            idx = self.get_indices(self.selected_channels, feature).astype(int) # features/columns indices (columns in .dat file) - e.g. max_correlation has 153 indices, SPLV has 1071 indices

            # CHECK AND FIX THIS CONDITION! LOOKS WEIRD.
            if (i == 0):
                segments_i, times, seizures_start, seizures_end = self.load_data(input_files[i])
                shape = np.shape(segments_i)
                segments = np.empty([shape[0], self.N_features])
            else:
                segments_i, _, _, _ = self.load_data(input_files[i])

            segments_i = segments_i[:, idx]
            shape = np.shape(segments_i)
            start = end
            end += shape[1]

            segments[:, start:end] = segments_i # concatenate all features per segment. E.g. if we use 2 feature files and each file have 153 feature columns, than the concatenation will result in 306 column features

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

    def annotate_data(self):
        # 0 - interictal
        # 1 - preictal
        # 2 - ictal
        annotations = np.zeros(len(self.times), dtype=int)

        discard_data = True
        discard_data_duration = 120 * 60  # 240 * 60 = 4 hours

        # preictal_file = open('./datasets/CHB-MIT/chb02/processed_data/preictal_segments.txt', 'w')
        # ictal_file = open('./datasets/CHB-MIT/chb02/processed_data/ictal_segments.txt', 'w')
        # interictal_file = open('./datasets/CHB-MIT/chb02/processed_data/interictal_segments.txt', 'w')

        preictal_list = []
        ictal_list = []
        interictal_list = []

        for index, seizure_start in enumerate(self.seizures_start):
            for i in range(0, len(self.times)):
                if(discard_data):
                    if (seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
                        annotations[i] = 1 # preictal
                        # preictal_file.write("%s\n" % self.segments[i])
                        preictal_list.append(self.segments[i])


                    elif(self.seizures_start[index] < self.times[i] and self.seizures_end[index] > self.times[i]):
                        annotations[i] = 2  # ictal
                        # ictal_file.write("%s\n" % self.segments[i])
                        ictal_list.append(self.segments[i])


                    elif(seizure_start > self.times[i] and ((seizure_start - self.preictal_duration) - self.times[i] < discard_data_duration) or ((self.seizures_end[index] + discard_data_duration) > self.times[i])): # discard_data_duration
                        # annotations[i] = 0 # interictal
                        # interictal_file.write("%s\n" % self.segments[i])
                        interictal_list.append(self.segments[i])

                else:
                    if (seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
                        annotations[i] = 1 # preictal
                    elif(self.seizures_start[index] < self.times[i] and self.seizures_end[index] > self.times[i]):
                        annotations[i] = 2  # ictal

        # preictal_file.close()
        # ictal_file.close()
        # interictal_file.close()

        np_preictal_file = np.asarray(preictal_list, dtype=np.float32)
        np_ictal_file = np.asarray(ictal_list, dtype=np.float32)
        np_interictal_file = np.asarray(interictal_list, dtype=np.float32)

        # writing

        parent_folder = './datasets/CHB-MIT/chb02/processed_data/preictal-' + str(int(self.preictal_duration/60))
        preictal_file = parent_folder + '/preictal_segments.npy'
        ictal_file = parent_folder + '/ictal_segments.npy'
        interictal_file = parent_folder + '/interictal_segments.npy'

        np.save(preictal_file, np_preictal_file)
        np.save(ictal_file, np_ictal_file)
        np.save(interictal_file, np_interictal_file)


        # Check how many data-points are annotated as 1 (preictal) or 2 (ictal) and 0 (interictal) respectively
        unique, counts = np.unique(annotations, return_counts=True)
        print('Total samples grouped by class (annotations: interictal=0, preictal=1, ictal=2): ', dict(zip(unique, counts)))

        return annotations # array filled with values 0, 1, 2


    # def annotate_data(self):
    #     # 0 - interictal
    #     # 1 - preictal
    #     # 2 - ictal
    #     annotations = np.zeros(len(self.times), dtype=int)
    #
    #     for index, seizure_start in enumerate(self.seizures_start):
    #         for i in range(0, len(self.times)):
    #             if (seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
    #                 annotations[i] = 1 # preictal
    #             elif(self.seizures_start[index] < self.times[i] and self.seizures_end[index] > self.times[i]):
    #                 annotations[i] = 2  # ictal
    #
    #     # Check how many data-points are annotated as 1 (preictal) or 2 (ictal) and 0 (interictal) respectively
    #     unique, counts = np.unique(annotations, return_counts=True)
    #     print('Total samples grouped by class (annotations: interictal=0, preictal=1, ictal=2): ', dict(zip(unique, counts)))
    #
    #     return annotations # array filled with values 0, 1, 2

    def k_fold_division(self, k):
        '''
           input : k to divide the set into k parts
        '''
        assign_to_set = np.zeros(len(self.annotations))

        indices = np.arange(0, len(self.segments))
        # ~ np.random.shuffle(indices)

        for i in range(0, int(len(indices) * (1 - 1. / k))):
            idx = indices[i]
            if (self.annotations[idx] == 0):
                assign_to_set[idx] = 1
            elif (self.annotations[idx] == 1):
                assign_to_set[idx] = 2

        for i in range(int(len(indices) * (1 - 1. / k)), len(indices)):
            idx = indices[i]
            if (self.annotations[idx] == 0):
                assign_to_set[idx] = 3
            elif (self.annotations[idx] == 1):
                assign_to_set[idx] = 4
        self.assign_sets(assign_to_set)

    def leave_one_seizure(self, i):
        '''
          input : i-th seizure to leave out (e.g. if there are 3 seizures in total, the second one will be out if i=1)
          This is used for Cross-Validation and leave 1 or the seizures
          out for cross-validation. Training set are indices of segments,
          same for validation set. Apart from seizure, a set without
          seizure is added (following the seizure)
        '''
        print('self.seizures_start: ', self.seizures_start)
        print('self.seizures_end: ', self.seizures_end)

        print('i-th seizure_start', self.seizures_start[i])
        print('i-th seizure_end', self.seizures_end[i])

        seizure_start = self.seizures_start[i]
        seizure_end = self.seizures_end[i]

        self.divide_data(seizure_start, seizure_end)

    def divide_data(self, test_seizure_start, test_seizure_end):
        '''
          input : seizure_start and seizure_end for seizure to keep for test set
          Make sure that all indices are in the preictal state before adding to
          test with seizure (test_ws)
          assign_to_set : 1 for train_wos, 2 train_ws, 3 test_wos, 4 test_ws
        '''
        assign_to_set = np.zeros(len(self.annotations))

        print("\nData preparation...")

        ###############################################################################################
        #################################### Test data with seizure ###################################
        ###############################################################################################
        test_start_ws = bisect.bisect_left(self.times, test_seizure_start - self.preictal_duration) # Location of the seizure (start_time - preictal_duration) including the preictal period (e.g. 30min) in times array. E.g. get 30 min (defined in self.preictal_duration) before the seizure started
        test_end_ws = bisect.bisect_left(self.times, test_seizure_start) # when seizure starts, it is the end of this 30 min duration before seizure

        # find all preictal annotated samples within this preictal duration period (e.g. 30min)
        # select N samples from preictal (e.g. 30min before seizure) and keep as test set WITH seizure (test_ws)
        length_ws = 0
        for idx in range(test_start_ws, test_end_ws):
            if (self.annotations[idx] == 1):
                assign_to_set[idx] = 4 # test_ws
                length_ws += 1

        print("1. Test WS finished")

        ###############################################################################################
        #################################### Training data with seizure ###############################
        ###############################################################################################
        seizures_start_left_training = [x for x in self.seizures_start if x != test_seizure_start]
        seizures_end_left_training = [x for x in self.seizures_end if x != test_seizure_end]

        print('seizures_start_left_training: ', seizures_start_left_training)
        print('seizures_end_left_training: ', seizures_end_left_training)

        for start_seizure_time, end_seizure_time in zip(seizures_start_left_training, seizures_end_left_training):

            train_start_ws = bisect.bisect_left(self.times, start_seizure_time - self.preictal_duration)
            train_end_ws = bisect.bisect_left(self.times, start_seizure_time)

            for idx in range(train_start_ws, train_end_ws):
                if (self.annotations[idx] == 1):
                    assign_to_set[idx] = 2  # train_ws

        print("2. Train WS finished")

        ###############################################################################################
        ######################### Discard some data before and after seizures #########################
        ###############################################################################################

        print('before discard seizures_start: ', self.seizures_start)
        print('before discard seizures_end: ', self.seizures_end)

        discard_data_criteria = 60 * 60 # 240 * 60 = 4 hours
        for seizure_start, seizure_end in zip(self.seizures_start, self.seizures_end):

            start_discard_before_seizure = bisect.bisect_left(self.times, seizure_start - (discard_data_criteria + self.preictal_duration))
            end_discard_before_seizure = bisect.bisect_left(self.times, seizure_start - self.preictal_duration)

            start_discard_after_seizure = bisect.bisect_right(self.times, seizure_end)
            end_discard_after_seizure = bisect.bisect_right(self.times, seizure_end + discard_data_criteria)

            before_seizure = np.arange(start_discard_before_seizure, end_discard_before_seizure)
            after_seizure = np.arange(start_discard_after_seizure, end_discard_after_seizure)
            mergedLists = np.concatenate((before_seizure , after_seizure), axis=None)

            for idx in mergedLists:
                if (assign_to_set[idx] == 0):
                    assign_to_set[idx] = 5

        print("3. Data discarding finished")

        ###############################################################################################
        ################################### Test data WITHOUT seizure #################################
        ###############################################################################################
        assign_to_set_zeros_indices = [i for i, j in enumerate(assign_to_set) if j == 0] # find all 0s to use for training
        exclude_test_wos_from_training = []

        # middle_of_list = int(len(assign_to_set_zeros_indices) / 2) + 6000
        # test_wos_N_indices = assign_to_set_zeros_indices[middle_of_list:middle_of_list+length_ws]
        test_wos_length = int(len(assign_to_set_zeros_indices) / 4)

        test_wos_N_indices = assign_to_set_zeros_indices[-test_wos_length:]

        test_data_length = 0
        for test_wos_index in test_wos_N_indices:
            if (assign_to_set[test_wos_index] == 0):
                assign_to_set[test_wos_index] = 3 # test_wos
                exclude_test_wos_from_training.append(test_wos_index)
            # if test_data_length == 2000:
            #     break
            # test_data_length += 1

        print("4. Test WOS finished")

        ###############################################################################################
        ################################# Training data WITHOUT seizure ###############################
        ###############################################################################################

        training_wos = [el for el in assign_to_set_zeros_indices if el not in exclude_test_wos_from_training]

        # train_wos_N_indices = training_wos[-2000:] # take training data from the end

        train_data_length = 0
        for train_wos_index in training_wos:
            if(assign_to_set[train_wos_index] == 0):
                assign_to_set[train_wos_index] = 1  # train_wos
            # if train_data_length == 1000:
            #     break
            # train_data_length += 1

                # training_wos_set_length += 1
            # if training_wos_set_length == 2000:
            #     break

        print("5. Train WOS finished")

        # Check how many data-points are assigned to training w/o seizure 1, training with seizure 2, test w/o seizure 3 and test with seizure 4
        unique, counts = np.unique(assign_to_set, return_counts=True)
        print('Total samples grouped by class (training and testing): ', dict(zip(unique, counts)))

        # assign_to_set = self.discard_data(assign_to_set)
        # unique, counts = np.unique(assign_to_set, return_counts=True)
        # print('Unique classes (after discarded samples): ', unique)
        # print('Total samples grouped by class (after discarded samples): ', dict(zip(unique, counts)))

        self.assign_sets(assign_to_set)

    def discard_data(self, assign_to_set):
        '''
        Remove segements 4h before interictal and 4h after
        input : assign_to_set with len(segments) indices from 1 to 4
        return: assign_to_set with idx of removed segments set to 0
        '''

        for i in range(0, len(self.seizures_start)):
            seizure_start = self.seizures_start[i]
            seizure_end = self.seizures_end[i]

            # select locations of samples from 4h before seizure happened up to preictal_duration (e.g. 30min) before seizure happened and discard all of them
            discard_prei_start = bisect.bisect_left(self.times, seizure_start - 240 * 60) # 240 * 60 = 4 hours
            discard_prei_end = bisect.bisect_left(self.times, seizure_start - self.preictal_duration)

            # select locations of all the samples just after the seizure end in a range/period of 4h and discard
            discard_posti_start = bisect.bisect_right(self.times, seizure_end)
            discard_posti_end = bisect.bisect_right(self.times, seizure_end + 240 * 60) # 240 * 60 = 4 hours

            for idx in range(discard_prei_start, discard_prei_end):
                assign_to_set[idx] = 0

            for idx in range(discard_posti_start, discard_posti_end):
                assign_to_set[idx] = 0

        return assign_to_set

    def assign_sets(self, assign_to_set):
        '''
          input : assign_to_set with len(segments) indices from 1 to 4
          assign_to_set : 1 for train_wos, 2 train_ws, 3 test_wos, 4 test_ws
          these four arrays hold only indices of segments placed in self.segments
        '''
        self.train_wos = []
        self.train_ws = []
        self.test_wos = []
        self.test_ws = []
        for idx in range(0, len(assign_to_set)):
            if (assign_to_set[idx] == 1):
                # from 23883 samples, take only 717 (len(train_ws)) + 2000 (just a number to make in total around 3000 samples)
                # so we won't have much more data for training than testing but somehow a proportion 70% training and 30% testing
                # if(len(self.train_wos) < (len(self.train_ws) + 2000)):
                #     self.train_wos.append(idx)

                self.train_wos.append(idx)
            if (assign_to_set[idx] == 2):
                self.train_ws.append(idx)
            if (assign_to_set[idx] == 3):
                self.test_wos.append(idx)
            if (assign_to_set[idx] == 4):
                self.test_ws.append(idx)
        print('len(self.train_wos): ', np.shape(self.train_wos))
        print('len(self.train_ws): ', np.shape(self.train_ws))
        print('len(self.test_wos): ', np.shape(self.test_wos))
        print('len(self.test_ws): ', np.shape(self.test_ws))

    # Function to calculate given percentage of a given number - helps when we use to split training and validation sets
    def percentage(self, percent, whole):
        return (percent * whole) / 100.0

    #################################################################################################################################
    ##################################################### START DATA BATCHING #######################################################
    #################################################################################################################################

    # print(np.shape(segments)) # (25355, 1377)
    # print(np.shape(train_wos)) # (2717,)
    # print(np.shape(train_ws)) # (717,)

    def X_training_batches_generator(self, segments, batch_size):
        # create the iterator and serve the batches
        sourceiter = iter(segments)
        while True:
            try:
                batchiter = islice(sourceiter, batch_size)
                yield chain([next(batchiter)], batchiter)
            except StopIteration:
                return

    # Get the segments using the indices lists
    def get_segments_using_indices_lists(self, segments, wos, ws):
        wos_segments = segments[wos] # wos indices
        ws_segments = segments[ws] # ws indices

        # print('WOS', np.shape(train_wos)) # WOS (2717, 1377)
        # print('WS', np.shape(train_ws)) # WS (717, 1377)

        # shuffle wos and ws segments before grouping them based on given num_input to avoid overfitting - this is randomly picking segments
        wos_shuffled = shuffle(wos_segments)
        ws_shuffled = shuffle(ws_segments)

        return wos_shuffled, ws_shuffled

    def segments_grouper(self, n_inputs, iterable, fillvalue=None):
        "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n_inputs
        fv = [fillvalue] * len(iterable[0])
        return zip_longest(fillvalue=list(fv), *args)

    def group_segments_to_create_input(self, training, testing):
        # group segments in num_inputs groups to form 1 input segment (e.g. one input/element of batch - batch[0])

        num_inputs = self.num_inputs # how many segments to take to create one input

        # Get segments
        # global var: self.segments = segments list holding all the loaded segments from the files
        # global var: self.train_wos = holding all the indices of segments w/o seizure
        # global var: self.train_ws = holding all the indices of segments with seizure
        val_wos_data = random.sample(list(self.train_wos), int(self.percentage(10, len(self.train_wos))))
        train_wos_data = [x for x in self.train_wos if x not in val_wos_data]

        val_ws_data = random.sample(list(self.train_ws), int(self.percentage(10, len(self.train_ws))))
        train_ws_data = [x for x in self.train_ws if x not in val_ws_data]

        # train_wos_segments, train_ws_segments = self.get_segments_using_indices_lists(self.segments, self.train_wos, self.train_ws)
        if(testing == True):
            wos_segments, ws_segments = self.get_segments_using_indices_lists(self.segments, self.test_wos, self.test_ws)
        else:
            if(training == True):
                # wos_segments, ws_segments = self.get_segments_using_indices_lists(self.segments, train_wos_data, train_ws_data)
                wos_segments, ws_segments = self.get_segments_using_indices_lists(self.segments, self.train_wos, self.train_ws)
            else:
                wos_segments, ws_segments = self.get_segments_using_indices_lists(self.segments, val_wos_data, val_ws_data)


        # Group segments
        # train_wos_segments_grouped = [train_wos_segments[n:n + num_inputs] for n in range(0, len(train_wos_segments), num_inputs)]
        # train_ws_segments_grouped = [train_ws_segments[n:n + num_inputs] for n in range(0, len(train_ws_segments), num_inputs)]

        wos_segments_grouped = self.segments_grouper(num_inputs, wos_segments, fillvalue=0)
        ws_segments_grouped = self.segments_grouper(num_inputs, ws_segments, fillvalue=0)

        wos_segments_grouped = list(wos_segments_grouped)
        ws_segments_grouped = list(ws_segments_grouped)

        # print(np.shape(train_wos_groups)) # (272, 10)
        # print(np.shape(train_ws_groups)) # (72, 10)

        return wos_segments_grouped, ws_segments_grouped

    def initialize_labels(self, training, testing):

        # get grouped segments - wos and ws
        wos_groups, ws_groups = self.group_segments_to_create_input(training, testing)

        # there is a label per group of segments (a.k.a input) - e.g. if one input is created from 10 segments, the input has one class 1 or 0
        # initialize label list interictal = 0
        y_wos = [[1,0]] * len(wos_groups)

        # initialize label list preictal = 1
        y_ws = [[0,1]] * len(ws_groups)

        return y_wos, y_ws, wos_groups, ws_groups

    def prepare_X_and_y_data(self, training, testing):

        # get initialized labels and also the training data/segments/inputs
        y_wos, y_ws, wos_groups, ws_groups = self.initialize_labels(training, testing)

        # Merge wos and ws data - ws being at the end of the new created list
        X_data = wos_groups + ws_groups
        # np.shape(X_training_data) (344)

        # Merge also the labels of wos and ws - with ws being at the end of new created list
        y_data = y_wos + y_ws
        # np.shape(y_training_data) (344, 2)

        # shuffle created inputs from multiple segments
        X, y = shuffle(X_data, y_data)
        # print(np.shape(X)) (344,)
        # print(np.shape(y)) (344, 2)

        # return training data and their labels ready to feed into network
        return X, y


    #################################################################################################################################
    ##################################################### END DATA BATCHING #########################################################
    #################################################################################################################################


    def train_next_batch(self, batch_size):
        '''
          input: size of the batch
          output: data used for training the algorithms
        '''
        # train_batch = np.empty([batch_size, 1, self.N_features, self.num_inputs]) # old
        train_batch = np.empty([batch_size, self.num_inputs, self.N_features]) #
        train_annotations = np.empty([batch_size, 2])
        for i in range(0, batch_size):
            p = random.random()
            if (p < 0.5):
                train_batch[i] = self.get_data(self.train_wos, self.num_inputs, self.segments_type_train)
                train_annotations[i] = [1, 0] # interictal
                # train_annotations[i] = 0 # interictal
            else:
                train_batch[i] = self.get_data(self.train_ws, self.num_inputs, self.segments_type_train)
                train_annotations[i] = [0, 1] # preictal
                # train_annotations[i] = 1 # preictal

        # print('train batch shape', train_batch.shape) # (32, 2)
        # print('anntoations shape: ', train_annotations.shape) # (32, 2)

        return train_batch, train_annotations

    def get_test_batch(self, batch_size):
        '''
          input: size of the batch
          output: data used for testing the algorithms
        '''
        # test_batch = np.empty([batch_size, 1, self.N_features, self.num_inputs])
        test_batch = np.empty([batch_size, self.num_inputs, self.N_features])
        test_annotations = np.empty([batch_size, self.num_classes])
        for i in range(0, batch_size):
            p = random.random()
            if (p < 0.5):
                test_batch[i] = self.get_data(self.test_wos, self.num_inputs, self.segments_type_test)
                test_annotations[i] = [1, 0] # interictal
                # test_annotations[i] = 0 # interictal
            else:
                test_batch[i] = self.get_data(self.test_ws, self.num_inputs, self.segments_type_test)
                test_annotations[i] = [0, 1] # preictal
                # test_annotations[i] = 1 # preictal
        return test_batch, test_annotations

    def get_data(self, idx_set, num_input, data_type):
        '''
        This method forms creates one input by selecting many segments from self.segments
        and creating one input in a batch.
        If batch of size 32 is chosen, there are needed 32 such inputs per batch.

        :param idx_set: the indices necessary to get segments from self.segments
        :param num_input: the number of segments (data points) to create an input
        :param data_type: sampling method
        :return:
        '''
        batch = np.empty([num_input, self.N_features])
        if (data_type == "random"):
            # This might be wrong! More than one batch can have save samples and that's not good!
            batch_indices = random.sample(idx_set, num_input)
        elif (data_type == "continuous"):
            data_continuous = False
            while (not data_continuous):
                idx = random.randrange(0, len(idx_set) - num_input)
                data_continuous = self.check_continuous(idx_set, idx, num_input)
            batch_indices = [idx_set[i] for i in range(idx, idx + num_input)]

        for i in range(0, num_input):
            batch[i, :] = self.segments[batch_indices[i]]

        # print('shape segments :', np.shape(self.segments))
        # print('type :', type(self.segments))
        # print('len :', len(self.segments[0]))
        # print(self.segments[0])

        return batch

    def check_continuous(self, idx_set, idx, num_input):
        if (idx_set[idx] + num_input == idx_set[idx + num_input]):
            return True
        else:
            return False

    def load_data(self, input_file):
        f = open(input_file, 'r')
        line = f.readline()
        seizures_start = []
        seizures_end = []
        seizures_time = [int(t) for t in line.strip().split(' ')]; # get the seizure times - first row/array in .dat files. This row is a pair of values indicating the start and end time of a seizure
        for i in range(0, int(len(seizures_time) / 2)): # /2 because if there are 3 seizures there are 6 values in seizure times (seizures_time) with start and end time for each seizure
            seizures_start.append(seizures_time[2 * i]) # append each seizure start time to seizures_start array. E.g if there are 6 times, then 0, 2, 4 are the starting times
            seizures_end.append(seizures_time[2 * i + 1]) # append each seizure end time to seizures_end array. E.g if there are 6 times, then 1, 3, 5 are the ending times

        lines = [line.rstrip().split(' ') for line in f]
        # N_features = len(lines[0]) - 1 # weird variable - not used and still is declared

        # segments = np.empty([len(lines), N_features + 1]) # commented because not used ---> initialize an empty-valued array that will hold all the segments (one segment = 5 sec long).
        segments = np.float32(lines) # assign segments with values from feature files (.dat)

        times = segments[:, 0] # store times (first column in .dat file)
        segments = segments[:, 1:] # store segments (all the columns excluding the first one (time column)) - one segment have many columns
        return segments, times, seizures_start, seizures_end


if __name__ == '__main__':
    main()