from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import multiprocessing as mp
import ctypes
import datetime
import time
import mne
import glob
import re
import numpy as np
import scipy
import scipy.stats
import random             
import matplotlib.pyplot as plt
import pywt
import networkx as nx

random.seed(0)

class EEG_data():
  def __init__(self, eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg):
    self.sampling = cfg.sampling
    self.duration = cfg.duration
    self.preictal_duration = cfg.preictal_duration
    self.sample_size=cfg.sampling*cfg.duration
    self.eeg_signals = eeg_signals
    self.eeg_start = eeg_start
    self.eeg_end = eeg_end
    self.seizures_start = seizures_start
    self.seizures_end = seizures_end
    # ~ self.get_features(1000)
    
  def get_signal(self,time):
    # ~ return self.eeg_signals.crop(time,time + duration)._data
    return self.eeg_signals.copy().crop(time,time + self.duration)
  
  def get_features(self, time):
    signal = self.get_signal(time)
    n_channels = len(signal)
    
    moments = self.compute_moments(signal)
    zero_crossings = self.get_zero_crossings(signal)
    peak_to_peak = self.get_peak_to_peak(signal)
    absolute_area = self.compute_absolute_area(signal)
    psd_ratio = self.power_spectral_density(signal)
    decorrelation_time = self.get_decorrelation_time(signal)
    dwt_coeffs = self.discrete_wavelet_transform(signal)
    max_correlation_coeffs = self.get_max_correlation(signal)
    
    EEG_graph = self.create_graph(max_correlation_coeffs)
    MST_graph = nx.minimum_spanning_tree(EEG_graph, weight='weight')
    graph_features = self.get_graph_features(MST_graph)
    
    features=moments.flatten() # 5xN
    features = np.hstack([features, zero_crossings.flatten()]) # 1xN
    features = np.hstack([features, peak_to_peak.flatten()]) # 1xN
    features = np.hstack([features, absolute_area.flatten()]) # 1xN
    features = np.hstack([features, psd_ratio.flatten()]) # 7xN
    features = np.hstack([features, decorrelation_time.flatten()]) # 1xN
    features = np.hstack([features, dwt_coeffs.flatten()]) # 8xN
    features = np.hstack([features, np.triu(max_correlation_coeffs).flatten()]) # N(N-1)/2
    features = np.hstack([features, graph_features])
    # ~ print(features.shape)
    return features
      
  def get_graph_features(self, MST_graph):
    betweenness = self.get_betweenness(MST_graph)
    clustering = self.get_clustering(MST_graph)
    eccentricity = self.get_eccentricity(MST_graph)
    local_efficiency = self.get_local_efficiency(MST_graph)
    global_efficiency = self.get_global_efficiency(MST_graph)
    diameter = self.get_diameter(MST_graph)
    radius = self.get_radius(MST_graph)
    average_shortest_path_length = self.get_average_shortest_path_length(MST_graph)
    
    graph_features=betweenness.flatten()
    graph_features = np.hstack([graph_features, clustering.flatten()])
    graph_features = np.hstack([graph_features, eccentricity.flatten()])
    graph_features = np.hstack([graph_features, local_efficiency])
    graph_features = np.hstack([graph_features, global_efficiency])
    graph_features = np.hstack([graph_features, diameter])
    graph_features = np.hstack([graph_features, radius])
    graph_features = np.hstack([graph_features, average_shortest_path_length])
    return graph_features
    
  def get_signal_channel(self, channel, time):
    return self.eeg_signals.copy().crop(time,time + self.duration)._data[channel]
  
  def compute_moments(self, signal):
    mean = np.mean (signal._data,1) 
    variance = np.var (signal._data,1)
    skewness = scipy.stats.skew(signal._data,1)
    kurtosis = scipy.stats.kurtosis(signal._data,1)
    return np.transpose(np.array([mean, variance, skewness, kurtosis, np.sqrt(variance)]))

  def get_zero_crossings(self, signal):
    n_channels = len(signal.ch_names)
    zero_crossings = np.zeros( [n_channels,1] )
    for channel in range(0,n_channels):
      zero_crossings[channel] = len(np.where(np.diff(np.sign( signal._data[channel,:])))[0])
    return zero_crossings
  
  def get_peak_to_peak(self, signal):
    n_channels = len(signal.ch_names)
    peak_to_peak = np.zeros( [n_channels,1] )
    for channel in range(0,n_channels):
      signal_max = np.max(signal._data[channel])
      signal_min = np.min(signal._data[channel])
      peak_to_peak[channel] = signal_max-signal_min
    return peak_to_peak
  
  def compute_absolute_area(self, signal):
    n_channels = len(signal.ch_names)
    absolute_areas = np.zeros( [n_channels,1] )
    for channel in range(0,n_channels):
      absolute_areas[channel] = np.sum (np.abs(signal._data[channel,:] ) ) * self.duration / len (signal._data[channel,:] )
    return absolute_areas
    
  def normalize_signals(self):
    for channel in range(0, len(self.eeg_signals.ch_names) ):
      self.eeg_signals._data[channel,:]=self.eeg_signals._data[channel,:]/np.max(self.eeg_signals._data[channel,:])
  
  def power_spectral_density(self, signal):
    n_channels = len(signal.ch_names)
    freqs_lb=[0,4,8,14,30,65,]
    freqs_ub=[3,7,13,30,55,110]
    psds, freqs = mne.time_frequency.psd_multitaper(signal, fmin=freqs_lb[0], fmax=freqs_ub[-1], picks = np.arange(0,n_channels), verbose=False)
    idx_lb = np.searchsorted(freqs,freqs_lb)
    idx_ub = np.searchsorted(freqs,freqs_ub)
    psd_channels = np.sum( psds, 1 )
    
    psd_ratio = np.zeros( [n_channels, len(idx_lb)+1] )
    
    #First total EEG energy
    for channel in range(0, n_channels ):
      sum_freq = np.sum( psds[channel,:] )
      psd_ratio[channel,0] = sum_freq
    
    for channel in range(0, n_channels ):
      for i in range(0, len(idx_lb)):
        sum_freq = np.sum( psds[channel, idx_lb[i]:idx_ub[i]] )
        ratio = sum_freq / psd_channels[channel]
        psd_ratio[channel,i+1] = ratio
    return psd_ratio
  
  def discrete_wavelet_transform(self, signal):
    n_channels = len(signal.ch_names)
    wavelet = pywt.Wavelet('db4')
    level=7
    coeffs = np.zeros( [n_channels, level+1] )
    decomposition = pywt.wavedec(signal._data, 'db4', mode='symmetric', level=level, axis=-1)
    for channel in range(0, n_channels ):
      for i in range(0,len(coeffs[channel])):
        coeffs[channel,i] = decomposition[i][channel][0]
    return coeffs
  
  def get_decorrelation_time(self, signal):
    n_channels = len(signal.ch_names)
    decorrelation_time = np.zeros( [n_channels, 1] )
    for channel in range(0, n_channels):
      decorr_idx=0;
      corr = scipy.correlate(signal._data[channel], signal._data[channel],"full")
      corr = np.roll(corr,len(signal._data[channel]))
      for i in range(0,len(corr)):
        if(corr[i] < 0):
          decorr_idx=i
          break
      decorrelation_time[channel] = decorr_idx / self.sampling
    return decorrelation_time
    
  def get_max_correlation(self, signal):
    n_channels = len(signal.ch_names)
    corr_coeffs = np.identity( n_channels )
    auto_corr_coeffs = self.get_auto_corr_coeffs(signal);
    for channel_1 in range(0, n_channels-1):
      for channel_2 in range(channel_1+1, n_channels):
        corr_coeffs[channel_1,channel_2] = self.channel_correlation(signal,auto_corr_coeffs, channel_1, channel_2)
        corr_coeffs[channel_2,channel_1] = corr_coeffs[channel_1,channel_2]
    return corr_coeffs
  
  def get_auto_corr_coeffs(self, signal):
    n_channels = len(signal.ch_names)
    auto_corr_coeffs=np.zeros(n_channels)
    for channel in range(0,n_channels):
      auto_corr_coeffs[channel] = scipy.correlate(signal._data[channel], signal._data[channel],"valid")
    return auto_corr_coeffs
    
  def channel_correlation(self, signal, auto_corr_coeffs, channel_1, channel_2):
    signal_1 = signal._data[channel_1];
    signal_2 = signal._data[channel_2];
    corr_1 = auto_corr_coeffs[channel_1]
    corr_2 = auto_corr_coeffs[channel_2]
    corr = scipy.correlate(signal_1, signal_2,"full")
    max_corr = np.max(corr)
    max_corr = max_corr / np.sqrt(corr_1*corr_2)
    return max_corr
  
  def to_array(self, dic):
    n = len(dic)
    arr = np.zeros([n,1])
    for key, value in dic.items():
      arr[key] = value
    return arr
    
  def get_betweenness(self, MST_graph):
    betweenness = nx.betweenness_centrality(MST_graph)
    return self.to_array(betweenness)
  
  def get_clustering(self, MST_graph):
    clustering = nx.clustering(MST_graph)
    return self.to_array(clustering)
    
  def get_eccentricity(self, MST_graph):
    eccentricity = nx.eccentricity(MST_graph)
    return self.to_array(eccentricity)
    
  def get_local_efficiency(self, MST_graph):
    local_efficiency = nx.local_efficiency(MST_graph)
    return local_efficiency
    # ~ return self.to_array(local_efficiency)
    
  def get_global_efficiency(self, MST_graph):
    global_efficiency = nx.global_efficiency(MST_graph)
    return global_efficiency
    # ~ return self.to_array(global_efficiency)
    
  def get_diameter(self, MST_graph):
    diameter = nx.diameter(MST_graph)
    return diameter
    # ~ return self.to_array(diameter)
      
  def get_radius(self, MST_graph):
    radius = nx.radius(MST_graph)
    return radius
    # ~ return self.to_array(radius)
      
  def get_average_shortest_path_length(self, MST_graph):
    average_shortest_path_length = nx.average_shortest_path_length(MST_graph)
    return average_shortest_path_length
    # ~ return self.to_array(average_shortest_path_length)
    
  def segment_signals(self):
    # we obtain N_segments signals of length duration
    N_segments = int(len(self.eeg_signals._data[0])/self.sample_size)
    self.segments_idx = np.linspace(0,len(self.eeg_signals._data[0]), N_segments, dtype=int, endpoint = False)[0:-1]
    self.segments_labels = [0] * N_segments
    
  def label_segments(self, next_seizure_time_start, next_seizure_time_end):
    for i in range( 0,len(self.segments_idx) ):
      time = self.eeg_start + self.segments_idx[i] / self.sampling
      if( next_seizure_time_start - time < self.preictal_duration and time < next_seizure_time_start):
        self.segments_labels[i]=1
      elif( next_seizure_time_start < time and time < next_seizure_time_end ):
        self.segments_labels[i]=2
        
  def get_segment_idx(self, i):
    idx=self.segments_idx[i]
    time = idx / self.sampling
    features = self.get_features(time)
    return features
    
  def get_segments(self, label):
    indices = [i for i, x in enumerate(self.segments_labels) if x==label]
    return self.segments[indices]

  def create_graph(self, corr_coeffs):
    n_channels = len(corr_coeffs)
    EEG_graph = nx.Graph()
    for channel in range(0, n_channels):
      EEG_graph.add_node(channel)
    for channel_1 in range(0, n_channels-1):
      for channel_2 in range(channel_1+1, n_channels):
        EEG_graph.add_edge(channel_1 ,channel_2 , weight = corr_coeffs[channel_1,channel_2])
    return EEG_graph
    
  def check(self,segment_idx, N_input):
    if(segment_idx+N_input >= len(self.segments_labels)):
      return False
    label = self.segments_labels[segment_idx]
    for i in range(0, N_input):
      if(label != self.segments_labels[segment_idx+i]):
        return False
    return True;
    
class Patient_data():
  
  def __init__(self, cfg):
    self.data_path = cfg.data_path
    self.patient_folder = cfg.patient_folder
    self.duration = cfg.duration
    self.sampling = cfg.sampling
    self.segments_type_train = cfg.segments_type_train
    self.segments_type_test = cfg.segments_type_test
    self.segments_type = cfg.segments_type
    self.timesteps = cfg.timesteps
    self.num_input = cfg.num_input
    self.num_classes = cfg.num_classes
    self.channels_names = cfg.channels_names
    self.eeg_data={}
    
    print(self.data_path+self.patient_folder)
    summary_filename = glob.glob(self.data_path+self.patient_folder+"/*summary.txt")[0]
    self.load_summary(summary_filename)
    self.current_time = 0
    self.current_day = 0
    self.load_files(cfg)
    
    train_keys, test_keys = self.divide_train_test()
    
    self.create_segments()
    self.interictal_train_list = self.get_list_segments(train_keys,0)
    self.preictal_train_list = self.get_list_segments(train_keys,1)
    self.interictal_test_list = self.get_list_segments(test_keys,0)
    self.preictal_test_list = self.get_list_segments(test_keys,1)
  
  def get_list_segments(self, keys, label):
    list_segments=[]
    for key in keys:
      for i in range(0, len(self.eeg_data[key].segments_idx) ):
        # ~ idx=self.eeg_data[key].segments_idx[i]
        lbl=self.eeg_data[key].segments_labels[i]
        if( label == lbl):
          list_segments.append([key,i])
    return list_segments
    
  def divide_train_test(self):
    # we ensure there is at least one file with a seizure
    # in the test set
    keys_with_seizures = self.get_keys_seizures()
    keys_no_seizures = self.get_keys_no_seizures()
    random.shuffle(keys_with_seizures)
    random.shuffle(keys_no_seizures)
    
    kws = max(1,int(0.8*len(keys_with_seizures)))
    kns = max(1,int(0.8*len(keys_no_seizures)))
    train_keys = keys_with_seizures[0:kws]
    train_keys += keys_no_seizures[0:kns]
    test_keys = keys_with_seizures[kws:]
    test_keys += keys_no_seizures[kns:]
    return tuple(train_keys), tuple(test_keys)
  
  def get_keys_seizures(self):
    keys=[]
    for key in self.eeg_data:
      if( len(self.eeg_data[key].seizures_start)>0 ):
        keys.append(key)
    return keys
    
  def get_keys_no_seizures(self):
    keys=[]
    for key in self.eeg_data:
      if( len(self.eeg_data[key].seizures_start)==0 ):
        keys.append(key)
    return keys
    
  def create_segments(self):
    for key in self.eeg_data:
      self.eeg_data[key].segment_signals()
    # TODO rewrite labeling part...
    for key in self.eeg_data:
      for s in  range(0,len(self.eeg_data[key].seizures_start)):
        next_seizure_time_start = self.eeg_data[key].seizures_start[s]
        next_seizure_time_end = self.eeg_data[key].seizures_end[s]
        for key_2 in self.eeg_data:
          self.eeg_data[key_2].label_segments(next_seizure_time_start, next_seizure_time_end)
  
  def add_train_preictal(self, mp_arr, batch_size, i, N_input=1):
    arr = np.frombuffer(mp_arr.get_obj())
    segments=arr.reshape( (batch_size,self.timesteps, N_input) )
    segments[i,:]= self.get_segments(self.preictal_train_list, N_input)
    
  def add_test_preictal(self, mp_arr, batch_size, i, N_input=1):
    arr = np.frombuffer(mp_arr.get_obj())
    segments=arr.reshape( (batch_size,self.timesteps, N_input) )
    segments[i,:]= self.get_segments(self.preictal_test_list, N_input)
    
  def add_train_interictal(self, mp_arr, batch_size, i, N_input=1):
    arr = np.frombuffer(mp_arr.get_obj())
    segments=arr.reshape( (batch_size,self.timesteps, N_input) )
    segments[i,:]= self.get_segments(self.interictal_train_list, N_input)
    
  def add_test_interictal(self, mp_arr, batch_size, i, N_input=1):
    arr = np.frombuffer(mp_arr.get_obj())
    segments=arr.reshape( (batch_size,self.timesteps, N_input) )
    segments[i,:]= self.get_segments(self.interictal_test_list, N_input)
  
  def get_segments(self, segments_list, N_input=1):
    if( self.segments_type == "continuous"):
      return self.get_segments_continuous(segments_list, N_input)
    elif( self.segments_type == "random"):
      return self.get_segments_random(segments_list, N_input)
    
  def get_segments_random(self, segments_list, N_input=1):
    idx_sets = random.sample(segments_list, N_input)
    segments=np.empty( (self.timesteps, N_input) )
    i=0
    for idx_set in idx_sets:
      data_key = idx_set[0]
      segment_idx = idx_set[1]
      segments[:,i]=self.eeg_data[ data_key ].get_segment_idx( segment_idx )
      i+=1
    return segments
    
  def get_segments_continuous(self, segments_list, N_input=1):
    idx = random.randrange( len(segments_list) - N_input )
    data_key = segments_list[idx][0]
    segment_idx = segments_list[idx][1]
    
    while( not self.eeg_data[data_key].check(segment_idx, N_input) ):
      idx = random.randrange( len(segments_list) - N_input )
      data_key = segments_list[idx][0]
      segment_idx = segments_list[idx][1]
    segments=np.empty( (self.timesteps, N_input) )
    for i in range(0,N_input):
      data_key = segments_list[idx+i][0]
      segment_idx = segments_list[idx+i][1]
      segments[:,i]=self.eeg_data[ data_key ].get_segment_idx( segment_idx )
    return segments
    
  def get_one_segment(self, mp_arr, idx_sets, i):
    data_key = idx_sets[i][0]
    segment_idx = idx_sets[i][1]
    arr = np.frombuffer(mp_arr.get_obj())
    segments=arr.reshape( (self.timesteps, int(len(arr)/self.timesteps)) )
    segments[:,i]=self.eeg_data[ data_key ].get_segment_idx( segment_idx )
      
  def train_next_batch(self,batch_size, num_input):
    self.segments_type = self.segments_type_train
    mp_batch_arr = mp.Array(ctypes.c_double, batch_size * self.timesteps * num_input)
    batch_arr = np.frombuffer(mp_batch_arr.get_obj())
    train_batch=batch_arr.reshape( (batch_size, self.timesteps, num_input) )
    
    train_annotation=np.zeros( (batch_size, self.num_classes) )
    processes = []
    
    preictal_ratio=0.5
    for i in range(0, batch_size):
      if(random.random()<preictal_ratio):
        p = mp.Process(target=self.add_train_preictal, args=( mp_batch_arr, batch_size, i, self.num_input))
        train_annotation[i] = [0,1]
      else:
        p = mp.Process(target=self.add_train_interictal, args=( mp_batch_arr, batch_size, i, self.num_input))
        train_annotation[i] = [1,0]
      processes.append(p)
    [x.start() for x in processes]
    [x.join() for x in processes]
    return train_batch, train_annotation
    
  def get_test_batch(self,batch_size, num_input):
    self.segments_type = self.segments_type_test
    mp_batch_arr = mp.Array(ctypes.c_double, batch_size * self.timesteps * num_input)
    batch_arr = np.frombuffer(mp_batch_arr.get_obj())
    test_batch=batch_arr.reshape( (batch_size, self.timesteps, num_input) )
    
    test_annotation=np.zeros( (batch_size, self.num_classes) )
    processes = []
    
    preictal_ratio=0.5
    for i in range(0, batch_size):
      if(random.random()<preictal_ratio):
        p = mp.Process(target=self.add_test_preictal, args=( mp_batch_arr, batch_size, i, self.num_input))
        test_annotation[i] = [0,1]
      else:
        p = mp.Process(target=self.add_test_interictal, args=( mp_batch_arr, batch_size, i, self.num_input))
        test_annotation[i] = [1,0]
      processes.append(p)
    [x.start() for x in processes]
    [x.join() for x in processes]
    return test_batch, test_annotation
    
  def load_summary(self, summary_filename):
    f=open(summary_filename)
    self.summary=f.readlines()
    f.close()
    
  def load_files(self, cfg):
    line_to_match="File Name:"
    indices = [index for index,line in enumerate(self.summary) if line_to_match in line]
    for idx in indices:
    # ~ for idx in indices[0:2]:
      filename = re.search('File Name: (.+?)\n', self.summary[idx] ).group(1)
      file_number = re.search('_(.+?).edf', self.summary[idx]).group(1)
      print("Loading: " + filename)
      self.eeg_data[file_number] = self.load_data(self.data_path+self.patient_folder+filename, idx, cfg)
    
  def load_data(self, filename, index_begin, cfg):
    # get info in the summary file
    file_number = re.search('_(.+?).edf', self.summary[index_begin]).group(1)
    
    eeg_start = re.search('Time: (.*?)\n', self.summary[index_begin+1]).group(1)
    if(eeg_start[0:2]=='24'):
      eeg_start=str.replace(eeg_start,'24','00')
    eeg_start = time.strptime(eeg_start,'%H:%M:%S')
    eeg_start = datetime.timedelta(hours=eeg_start.tm_hour,minutes=eeg_start.tm_min,seconds=eeg_start.tm_sec).total_seconds()
    eeg_end = re.search('Time: (.*?)\n', self.summary[index_begin+2]).group(1)
    if(eeg_end[0:2]=='24'):
      eeg_end=str.replace(eeg_end,'24','00')
    eeg_end = time.strptime(eeg_end,'%H:%M:%S')
    eeg_end = datetime.timedelta(hours=eeg_end.tm_hour,minutes=eeg_end.tm_min,seconds=eeg_end.tm_sec).total_seconds()
    
    # next day
    if(eeg_start<self.current_time):
      self.current_day=self.current_day+1
    self.current_time = eeg_start
    eeg_start = int(eeg_start + 24*3600*self.current_day)
    eeg_end = int(eeg_end + 24*3600*self.current_day)
    n_seizures =  int(re.search(':(.+?)\n', self.summary[index_begin+3]).group(1))
    seizures_start = [0]*n_seizures
    seizures_end = [0]*n_seizures
    # load the seizures info in the summary file
    for i in range(0,n_seizures):
      idx=index_begin+4+2*i
      seizures_start[i] = int(re.search(':(.+?) seconds\n', self.summary[idx]).group(1))+eeg_start
      seizures_end[i] = int(re.search(':(.+?) seconds\n', self.summary[idx+1]).group(1))+eeg_start
    eeg_signals=mne.io.read_raw_edf(filename, preload=True, stim_channel=None)
    
    eeg_signals.pick_channels(self.channels_names)
    # return the EEG signal
    return EEG_data(eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg)
  
  def print_info(self):
    print("Number of files loaded: " + str(len(self.eeg_data)) )
