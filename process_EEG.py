from __future__ import print_function
# import tensorflow as tf
# from tensorflow.contrib import rnn
import multiprocessing as mp
import ctypes
import datetime
import time
import mne
import glob
import re
import sys
import numpy as np
import scipy
import scipy.stats
import scipy.signal
from sklearn.neighbors import NearestNeighbors
import random
import matplotlib.pyplot as plt
import pywt
import networkx as nx

sys.path.insert(0, "./cpp/")
import dstl


class Config():
    '''
    Config class, init take data folder as well as patient number
    the rest of the parameters is to determine which feature to compute,
    which channels to take into account, which frequencies to sample,
    and where to store the features.
    '''

    def __init__(self, data_path="./datasets/CHB-MIT/", patient=2):
        self.patient = patient
        self.patient_folder = 'chb{:02d}/'.format(patient)
        self.output_dir = "./data_features/CHB-MIT_features/" + self.patient_folder + "/"
        self.data_path = data_path
        # data parameters
        self.sampling = 256  # Hz, specified by data
        self.duration = 5  # duration of samples to train in seconds
        self.freqs = np.array([[0.1, 4], [4, 7], [7, 13], [13, 15], [14, 30], [30, 45], [65, 120]])
        self.channels_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3",
                               "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                               "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2", "FZ-CZ", "CZ-PZ"]
        self.N_channels = len(self.channels_names)
        # ~ self.features_name = ["max_correlation","nonlinear_interdependence","DSTL","SPLV"]
        # self.features_name = ["SPLV"]
        self.features_name = ["max_correlation","nonlinear_interdependence","DSTL","SPLV","univariate"]
        # self.features_name = ["univariate"]
        # self.features_name = ["DSTL"]
        # ~ self.features_name = ["max_correlation"]
        self.features_len = {}
        self.features_len["max_correlation"] = int(self.N_channels * (self.N_channels - 1) / 2.)
        self.features_len["nonlinear_interdependence"] = int(self.N_channels * (self.N_channels - 1) / 2.)
        self.features_len["DSTL"] = int(self.N_channels * (self.N_channels - 1) / 2.)
        self.features_len["SPLV"] = int(self.N_channels * (self.N_channels - 1) / 2. * len(self.freqs))
        self.features_len["univariate"] = int(self.N_channels * 24)
        print('Length features univariate: ', self.features_len["univariate"])
        self.N_features = 0

        for key in self.features_name:
            self.N_features += self.features_len[key]


class EEG_data():
    def __init__(self, eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg):

        self.sampling = cfg.sampling
        self.duration = cfg.duration
        self.sample_size = cfg.sampling * cfg.duration
        self.N_channels = cfg.N_channels
        self.eeg_signals = eeg_signals
        self.eeg_start = eeg_start
        self.eeg_end = eeg_end
        self.seizures_start = seizures_start
        self.seizures_end = seizures_end
        self.eeg_signals._data = self.filter_power(self.eeg_signals)
        self.eeg_signals._data = self.filter_low_high(self.eeg_signals)
        self.N_features = cfg.N_features
        self.features_len = cfg.features_len
        self.N_channels = cfg.N_channels
        self.freqs = cfg.freqs

    def get_signal(self, time):
        return self.eeg_signals.copy().crop(time, time + self.duration)

    def get_feature(self, feature_name, time):
        signal = self.get_signal(time)
        if (feature_name == "max_correlation"):
            feature = self.get_max_correlation(signal)
        elif (feature_name == "nonlinear_interdependence"):
            feature = self.nonlinear_interdependence(signal)
        elif (feature_name == "DSTL"):
            feature = self.get_DSTL(signal)
        elif (feature_name == "SPLV"):
            feature = self.get_SPLV(signal)
        elif (feature_name == "univariate"):
            feature = self.get_univariate_features(signal)
        else:
            print("Error in cfg.features_name")
        return feature

    def get_all_features(self, feature_name):
        ''' Computes all the features corresponding to feature_name in parallel
        input: a string with the name of the feature to compute
        output: features corresponding to each segments
        '''
        N_segments = len(self.segments_idx)

        mp_batch_arr = mp.Array(ctypes.c_double, N_segments * (self.features_len[feature_name] + 1))
        batch_arr = np.frombuffer(mp_batch_arr.get_obj())
        features = batch_arr.reshape((N_segments, self.features_len[feature_name] + 1))
        processes = []

        for i in range(0, N_segments):
            time = self.segments_idx[i] / self.sampling
            features[i, 0] = time + self.eeg_start
            p = mp.Process(target=self.add_features, args=(mp_batch_arr, feature_name, N_segments, time, i))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]
        return features

    def add_features(self, mp_arr, feature_name, N_segments, time, i):
        arr = np.frombuffer(mp_arr.get_obj())
        segments = arr.reshape((N_segments, self.features_len[feature_name] + 1))
        segments[i, 1] = time
        segments[i, 1:] = self.get_feature(feature_name, time) # 25355

    def segment_signals(self):
        ''' divides data into N_segments signals of length sample_size specified in config
        '''
        N_segments = int(len(self.eeg_signals._data[0]) / self.sample_size)
        self.segments_idx = np.linspace(0, len(self.eeg_signals._data[0]), N_segments, dtype=int, endpoint=False)[0:-1]
        print('N_segments (line 137 @ process_EEG.py)', N_segments)
        print('self.sample_size', self.sample_size)
        print('self.eeg_signals._data[0]', len(self.eeg_signals._data[0]))
        print('self.segments_idx', len(self.segments_idx))


    def get_univariate_features(self, signal):
        '''
          Computes various univariate features from a signal.
          The description of the features can be found in:
          Tsiouris, et al. "A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals
          ." Computers in biology and medicine 99 (2018): 24-37.
          zero crossings was removed as it made the neural networks unable to converge
        '''
        moments = self.compute_moments(signal)
        zero_crossings = self.get_zero_crossings(signal)
        peak_to_peak = self.get_peak_to_peak(signal)
        absolute_area = self.compute_absolute_area(signal)
        psd_ratio = self.power_spectral_density(signal)
        decorrelation_time = self.get_decorrelation_time(signal)
        dwt_coeffs = self.discrete_wavelet_transform(signal)

        features = moments.flatten()  # 5xN
        # ~ features = np.hstack([features, zero_crossings.flatten()]) # 1xN
        features = np.hstack([features, peak_to_peak.flatten()])  # 1xN
        features = np.hstack([features, absolute_area.flatten()])  # 1xN
        features = np.hstack([features, psd_ratio.flatten()])  # 8xN
        features = np.hstack([features, decorrelation_time.flatten()])  # 1xN
        features = np.hstack([features, dwt_coeffs.flatten()])  # 8xN
        return features

    def get_max_correlation(self, signal):
        ''' Computes the maximum of correlations between the channels of a signal
        input: EEG signal in RawEDF format
        output: correlation matrix (1D)
        '''
        n_channels = len(signal.ch_names)
        corr_coeffs = np.empty(int(n_channels * (n_channels - 1) / 2))
        auto_corr_coeffs = self.get_auto_corr_coeffs(signal);
        idx = 0
        for channel_a in range(0, n_channels - 1):
            for channel_b in range(channel_a + 1, n_channels):
                corr_coeffs[idx] = self.channel_correlation(signal, auto_corr_coeffs, channel_a, channel_b)
                idx += 1
        return corr_coeffs

    def get_auto_corr_coeffs(self, signal):
        n_channels = len(signal.ch_names)
        auto_corr_coeffs = np.zeros(n_channels)
        for channel in range(0, n_channels):
            auto_corr_coeffs[channel] = scipy.correlate(signal._data[channel], signal._data[channel], "valid")
        return auto_corr_coeffs

    def channel_correlation(self, signal, auto_corr_coeffs, channel_1, channel_2):
        signal_1 = signal._data[channel_1];
        signal_2 = signal._data[channel_2];
        corr_1 = auto_corr_coeffs[channel_1]
        corr_2 = auto_corr_coeffs[channel_2]
        corr = scipy.correlate(signal_1, signal_2, "full")
        max_corr = np.max(corr)
        max_corr = max_corr / np.sqrt(corr_1 * corr_2)
        return max_corr

    def nonlinear_interdependence(self, signal):
        ''' Compute the nonlinear interdependance between all channels of a signal
        input: signal in RawEDF format
        output: noninear interdependance matrix (1D)
        '''
        n_channels = len(signal.ch_names)
        embedded_signals = self.embed_signals(signal)
        full_S = self.get_full_S(embedded_signals)
        S = np.empty(int(n_channels * (n_channels - 1) / 2))
        idx = 0
        for channel_1 in range(0, n_channels):
            for channel_2 in range(channel_1 + 1, n_channels):
                S[idx] = (full_S[channel_1, channel_2] + full_S[channel_2, channel_1]) / 2
                idx += 1
        return S

    def embed_signals(self, signal, d=10, lag=6):
        ''' embed a signal in d dimension, with lag delay
        input: signal in RawEDF format, d dimension and lag delay
        output: embedded signal in numpy format
        '''
        n_channels = len(signal.ch_names)
        len_signal = len(signal._data[0])
        embedded_signals = np.empty([n_channels, len_signal - (d - 1) * lag, d])
        for channel in range(0, n_channels):
            for i in range(0, len_signal - (d - 1) * lag):
                for j in range(0, d):
                    idx = i - j * lag + (d - 1) * lag
                    embedded_signals[channel, i, j] = signal._data[channel][idx]
        return embedded_signals

    def get_full_S(self, embedded_signals):
        ''' Computes the nonlinear interdependance matrix
        input: embedded signals in numpy foramt
        output: S matrix
        '''
        n_channels = len(embedded_signals)
        full_S = np.empty([n_channels, n_channels])
        for channel_1 in range(0, n_channels):
            for channel_2 in range(0, n_channels):
                if (channel_1 != channel_2):
                    full_S[channel_1, channel_2] = self.get_S(embedded_signals[channel_1, :, :],
                                                              embedded_signals[channel_2, :, :])
        return full_S

    def get_S(self, xa, xb):
        ''' Compute the nonlinear interdependance between two signals
        inputs: embedded signals in numpy format
        output: nonlinear interdependance value
        '''
        K = 5
        nbrs_a = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree', metric='euclidean').fit(xa)
        nbrs_b = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree', metric='euclidean').fit(xb)
        distances_a, indices_a = nbrs_a.kneighbors(xa)
        distances_b, indices_b = nbrs_b.kneighbors(xb)

        S = 0
        for t in range(0, len(xa)):
            ra = np.average(np.square(distances_a[t, 1:]))
            rab = 0
            for k in range(0, K):
                rab += np.sum(np.square(xa[t] - xa[indices_b[t][k + 1]]))
            rab = rab / (K)
            S += ra / rab
        S = S / len(xa)
        return S

    def get_DSTL(self, signal):
        '''
        Compute the DSTL of a signal
        Input: signal in RawEDF format
        Output DSTL matrix
        Most parts are in C++, as loops are not efficient in Python
        '''
        dt = 12
        n_channels = len(signal.ch_names)
        embedded_signals = self.embed_signals(signal, d=7, lag=6)
        print(np.isinf(np.sum(embedded_signals)))
        DSTL = dstl.stlm(embedded_signals)
        # ~ STLm = np.zeros(n_channels)
        # ~ N = len(embedded_signals[0])
        # ~ for channel in range(0,n_channels):
        # ~ print(channel)
        # ~ for i in range(0,N-dt):
        # ~ print(i, N-dt)
        # ~ d0, d_dt = self.get_perturbations(embedded_signals[channel], i, dt)
        # ~ STLm[channel] += np.log2(d0/d_dt)
        # ~ STLm[channel] *= 1./((N-dt)*dt)
        # ~ DSTL = np.zeros(int(n_channels * (n_channels-1) /2))
        # ~ idx=0
        # ~ for channel_a in range(0,n_channels-1):
        # ~ for channel_b in range(channel_a+1,n_channels):
        # ~ DSTL[channel_a,channel_b] = STLm[channel_a,channel_b]
        # ~ idx += 1
        return DSTL

    def get_perturbations(self, embedded_signal, i, dt):
        '''
        Find the transverse in the phase space. This function has
        been implemented in C++, as it is much faster.
        '''
        j = self.get_transverse(embedded_signal, i, dt)
        if (j == -1):
            print("not found")
        d_0 = np.linalg.norm(embedded_signal[i] - embedded_signal[j])
        d_dt = np.linalg.norm(embedded_signal[i + dt] - embedded_signal[j + dt])
        return d_0, d_dt

    def get_transverse(self, embedded_signal, i, dt):
        '''
        see C++ code
        '''
        V = 0.1
        b = 0.05
        c = 0.1
        lag = 6
        d = 7
        IDIST3 = (d - 1) * lag
        delta = self.get_delta(embedded_signal, lag, IDIST3, i)
        dist = 0

        found_transverse = False
        found_angle = False
        while (True):
            print(c, V)
            for t in range(IDIST3, len(embedded_signal)):
                if (i + t + dt < len(embedded_signal)):
                    Vij = self.angle(embedded_signal[i], embedded_signal[i + t])
                    if (Vij < V):
                        dist = np.linalg.norm(embedded_signal[i, :] - embedded_signal[i + t, :])
                        if (dist < 0.5 * delta):
                            found_angle = True
                        if (dist > b * delta and dist < c * delta):
                            return i + t
                if (i - t > 0):
                    Vij = self.angle(embedded_signal[i], embedded_signal[i - t])
                    if (Vij < V):
                        dist = np.linalg.norm(embedded_signal[i, :] - embedded_signal[i - t, :])
                        if (dist < 0.5 * delta):
                            found_angle = True
                        if (dist > b * delta and dist < c * delta):
                            return i - t
            print(found_angle, Vij, delta, dist)
            if (not found_angle and V <= 0.91):
                V += 0.1
            elif (c >= 0.49 and V >= 0.99):
                return -1
            else:
                c += 0.1

    def get_delta(self, embedded_signal, tau, IDIST3, i):
        '''
        see C++ code
        '''
        delta = 0
        for t in range(tau, IDIST3):
            if (i + t < len(embedded_signal)):
                delta = max(delta, np.linalg.norm(embedded_signal[i, :] - embedded_signal[i + t, :]))
            if (i - t > 0):
                delta = max(delta, np.linalg.norm(embedded_signal[i, :] - embedded_signal[i - t, :]))
        return delta

    def unit_vector(self, vector):
        '''
        see C++ code
        '''
        return vector / np.linalg.norm(vector)

    def angle(self, x1, x2):
        '''
        Compute angle between two n-vectors
        '''
        x1_u = self.unit_vector(x1)
        x2_u = self.unit_vector(x2)
        return abs(np.arccos(np.clip(np.dot(x1_u, x2_u), -1.0, 1.0)))

    def get_SPLV(self, signals):
        ''' Computes phase-locking synchrony between all channels.
        Note that we use hilbert transform rather than Gabor, as
        suggested in
        Mirowski, Piotr, et al. "Classification of patterns of EEG synchronization for seizure prediction." Clinical neurophysiology 120.11 (2009): 1927-1940.
        Input: signal in RawEDF format
        Output: SPLV matrix
        '''
        n_channels = len(signals._data)
        filtered_signals = self.filter_signal(signals)
        hilbert_signals = self.get_hilbert_signals(filtered_signals)
        SPLV = np.empty(int(n_channels * (n_channels - 1) / 2) * len(hilbert_signals))
        idx = 0
        for f in range(0, len(hilbert_signals)):
            for channel_a in range(0, n_channels - 1):
                for channel_b in range(channel_a + 1, n_channels):
                    SPLV[idx] = self.PLV(hilbert_signals[f], channel_a, channel_b)
                    idx += 1
        return SPLV

    def PLV(self, hilbert_signal, channel_a, channel_b):
        '''
        Computes phase-locking synchrony between all channels.
        inputs: Hilbert transform of a signal, and integer number of the considered channels
        output: PLV value
        '''
        phase_a = np.unwrap(np.angle(hilbert_signal[channel_a, :]))
        phase_b = np.unwrap(np.angle(hilbert_signal[channel_b, :]))
        PLV = np.exp(1j * (phase_a - phase_b))
        return np.absolute(np.sum(PLV) / len(PLV))

    def filter_signal(self, signals):
        ''' input : RawEDF signal
            Output : array of len(freq) signals, filtered by fir
            the ntaps is defined as ~4 observations. The signals are filtered between frequencies defined in config file.
        '''
        filtered_signals = [None] * len(self.freqs)
        f = 0
        for freq in self.freqs:
            filtered_signals[f] = [None] * len(signals._data)
            avg_freq = np.average(freq)
            t_ms = 1000 / avg_freq
            ntaps = int(self.sampling * t_ms / 1000 * 4)
            # ~ ntaps = 1000
            fir = scipy.signal.firwin(ntaps, [freq[0], freq[1]], width=2, pass_zero=False, nyq=self.sampling / 2.)
            filtered_signals[f] = scipy.signal.convolve(signals._data, fir[np.newaxis, :], mode='valid')
            f += 1
        return filtered_signals

    def get_hilbert_signals(self, filtered_signals):
        hilbert_signals = [None] * len(filtered_signals)
        for f in range(0, len(filtered_signals)):
            hilbert_signals[f] = np.empty(filtered_signals[f].shape)
            for channel in range(0, len(filtered_signals[f])):
                hilbert_signals[f][channel, :] = np.imag(scipy.signal.hilbert(filtered_signals[f][channel, :]))
        return hilbert_signals

    def filter_low_high(self, signals):
        ''' Input : signals in RawEDF format
            Output : filtered signals
            Note that we only remove frequencies above 120Hz, but we could remove low frequence
            up to 0.5Hz, but it seems to add a lot of error to the signal
        '''
        low_freq = 120 * 2 / self.sampling
        # ~ high_freq=0.5*2/self.sampling
        b_low, a_low = scipy.signal.butter(1, low_freq, btype='lowpass')
        # ~ b_high, a_high = scipy.signal.butter(5, high_freq, btype='highpass')
        filtered_signals = scipy.signal.filtfilt(b_low, a_low, signals._data)
        # ~ filtered_signals=scipy.signal.filtfilt(b_high, a_high, filtered_signals)
        return filtered_signals

    def filter_power(self, signals):
        ''' Input : signals in RawEDF format
            Output : filtered signals
            Remove the Power line frequencies between 59 and 61 Hz
        '''
        freqs = np.array([59, 61]) * 2 / self.sampling
        order = 5
        b, a = scipy.signal.butter(order, freqs, btype='bandstop')
        return scipy.signal.filtfilt(b, a, signals._data)

    def compute_moments(self, signal):
        mean = np.mean(signal._data, 1)
        variance = np.var(signal._data, 1)
        skewness = scipy.stats.skew(signal._data, 1)
        kurtosis = scipy.stats.kurtosis(signal._data, 1)
        return np.transpose(np.array([mean, variance, skewness, kurtosis, np.sqrt(variance)]))

    def get_zero_crossings(self, signal):
        n_channels = len(signal.ch_names)
        zero_crossings = np.zeros([n_channels, 1])
        for channel in range(0, n_channels):
            zero_crossings[channel] = len(np.where(np.diff(np.sign(signal._data[channel, :])))[0])
        return zero_crossings

    def get_peak_to_peak(self, signal):
        n_channels = len(signal.ch_names)
        peak_to_peak = np.zeros([n_channels, 1])
        for channel in range(0, n_channels):
            signal_max = np.max(signal._data[channel])
            signal_min = np.min(signal._data[channel])
            peak_to_peak[channel] = signal_max - signal_min
        return peak_to_peak

    def compute_absolute_area(self, signal):
        n_channels = len(signal.ch_names)
        absolute_areas = np.zeros([n_channels, 1])
        for channel in range(0, n_channels):
            absolute_areas[channel] = np.sum(np.abs(signal._data[channel, :])) * self.duration / len(
                signal._data[channel, :])
        return absolute_areas

    def normalize_signals(self):
        for channel in range(0, len(self.eeg_signals.ch_names)):
            self.eeg_signals._data[channel, :] = self.eeg_signals._data[channel, :] / np.max(
                self.eeg_signals._data[channel, :])

    def power_spectral_density(self, signal):
        n_channels = len(signal.ch_names)
        freqs_lb = self.freqs[:, 0]
        freqs_ub = self.freqs[:, 1]
        psds, freqs = mne.time_frequency.psd_multitaper(signal, fmin=freqs_lb[0], fmax=freqs_ub[-1],
                                                        picks=np.arange(0, n_channels), verbose=False)
        idx_lb = np.searchsorted(freqs, freqs_lb)
        idx_ub = np.searchsorted(freqs, freqs_ub)
        psd_channels = np.sum(psds, 1)

        psd_ratio = np.zeros([n_channels, len(idx_lb) + 1])

        # First total EEG energy
        for channel in range(0, n_channels):
            sum_freq = np.sum(psds[channel, :])
            psd_ratio[channel, 0] = sum_freq

        for channel in range(0, n_channels):
            for i in range(0, len(idx_lb)):
                sum_freq = np.sum(psds[channel, idx_lb[i]:idx_ub[i]])
                ratio = sum_freq / psd_channels[channel]
                psd_ratio[channel, i + 1] = ratio
        return psd_ratio

    def get_decorrelation_time(self, signal):
        n_channels = len(signal.ch_names)
        decorrelation_time = np.zeros([n_channels, 1])
        for channel in range(0, n_channels):
            decorr_idx = 0;
            corr = scipy.correlate(signal._data[channel], signal._data[channel], "full")
            corr = np.roll(corr, len(signal._data[channel]))
            for i in range(0, len(corr)):
                if (corr[i] < 0):
                    decorr_idx = i
                    break
            decorrelation_time[channel] = decorr_idx / self.sampling
        return decorrelation_time

    def discrete_wavelet_transform(self, signal):
        n_channels = len(signal.ch_names)
        wavelet = pywt.Wavelet('db4')
        level = 7
        coeffs = np.zeros([n_channels, level + 1])
        decomposition = pywt.wavedec(signal._data, 'db4', mode='symmetric', level=level, axis=-1)
        for channel in range(0, n_channels):
            for i in range(0, len(coeffs[channel])):
                coeffs[channel, i] = decomposition[i][channel][0]
        return coeffs


class Patient_data():
    '''
    Class to load all the files related to a patient
    input: config fig
    output: writes all the requested feature in cfg.output_dir
    '''

    def __init__(self, cfg):
        self.data_path = cfg.data_path
        self.patient_folder = cfg.patient_folder
        self.output_dir = cfg.output_dir
        self.duration = cfg.duration
        self.sampling = cfg.sampling
        self.channels_names = cfg.channels_names
        self.eeg_data = {}
        self.features_name = cfg.features_name
        self.features_len = cfg.features_len

        print(self.data_path + self.patient_folder)
        summary_filename = glob.glob(self.data_path + self.patient_folder + "/*summary.txt")[0]
        self.seizures_time = []
        self.load_summary(summary_filename)
        self.current_time = 0
        self.current_day = 0
        self.load_files(cfg)
        self.create_segments()
        for feature_name in self.features_name:
            self.save_segments(feature_name)

    def save_segments(self, feature_name):
        f = open(self.output_dir + "/" + feature_name + ".dat", 'w')
        self.write_seizures(f)
        for key in self.eeg_data:
            segments = self.eeg_data[key].get_all_features(feature_name)
            self.write_segments(segments, f)

    def write_segments(self, segments, f):
        for segment in segments:
            f.write(" ".join(str(x) for x in segment))
            f.write('\n')

    def write_seizures(self, f):
        for st in self.seizures_time:
            f.write(str(st[0]) + ' ' + str(st[1]) + ' ')
        f.write('\n')

    def create_segments(self):
        print('self.eeg_data in line 591 @ process_EEG.py: ', self.eeg_data)
        for key in self.eeg_data:
            self.eeg_data[key].segment_signals()

    def load_summary(self, summary_filename):
        f = open(summary_filename)
        self.summary = f.readlines()
        f.close()

    def load_files(self, cfg):
        line_to_match = "File Name:"
        indices = [index for index, line in enumerate(self.summary) if line_to_match in line]
        for idx in indices:
            # ~ for idx in indices[2:3]:
            filename = re.search('File Name: (.+?)\n', self.summary[idx]).group(1)
            file_number = re.search('_(.+?).edf', self.summary[idx]).group(1)
            print("Loading: " + filename)
            self.eeg_data[file_number] = self.load_data(self.data_path + self.patient_folder + filename, idx, cfg)
        print('self.eeg_data shape: ', len(self.eeg_data))

    def load_data(self, filename, index_begin, cfg):
        # get info in the summary file
        file_number = re.search('_(.+?).edf', self.summary[index_begin]).group(1)

        eeg_start = re.search('Time: (.*?)\n', self.summary[index_begin + 1]).group(1)
        if (eeg_start[0:2] == '24'):
            eeg_start = str.replace(eeg_start, '24', '00')
        eeg_start = time.strptime(eeg_start, '%H:%M:%S')
        eeg_start = datetime.timedelta(hours=eeg_start.tm_hour, minutes=eeg_start.tm_min,
                                       seconds=eeg_start.tm_sec).total_seconds()
        eeg_end = re.search('Time: (.*?)\n', self.summary[index_begin + 2]).group(1)
        if (eeg_end[0:2] == '24'):
            eeg_end = str.replace(eeg_end, '24', '00')
        eeg_end = time.strptime(eeg_end, '%H:%M:%S')
        eeg_end = datetime.timedelta(hours=eeg_end.tm_hour, minutes=eeg_end.tm_min,
                                     seconds=eeg_end.tm_sec).total_seconds()

        # next day
        if (eeg_start < self.current_time):
            self.current_day = self.current_day + 1
        self.current_time = eeg_start
        eeg_start = int(eeg_start + 24 * 3600 * self.current_day)
        eeg_end = int(eeg_end + 24 * 3600 * self.current_day)
        n_seizures = int(re.search(':(.+?)\n', self.summary[index_begin + 3]).group(1))
        seizures_start = [0] * n_seizures
        seizures_end = [0] * n_seizures
        # load the seizures info in the summary file
        for i in range(0, n_seizures):
            idx = index_begin + 4 + 2 * i
            seizures_start[i] = int(re.search(':(.+?) seconds\n', self.summary[idx]).group(1)) + eeg_start
            seizures_end[i] = int(re.search(':(.+?) seconds\n', self.summary[idx + 1]).group(1)) + eeg_start
            self.seizures_time.append([seizures_start[i], seizures_end[i]])

        eeg_signals = mne.io.read_raw_edf(filename, preload=True, stim_channel=None)
        eeg_signals.pick_channels(self.channels_names)
        # return the EEG signal
        return EEG_data(eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg)

    def print_info(self):
        print("Number of files loaded: " + str(len(self.eeg_data)))


cfg = Config(data_path="/home/jakupi/datasets/CHB-MIT/", patient=2)
patient_data = Patient_data(cfg)
