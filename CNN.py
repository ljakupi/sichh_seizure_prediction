from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib 


def CNN(x, cfg):
  ''' create a neural network corresponding to the one in in
  Mirowski, Piotr, et al. "Classification of patterns of EEG synchronization for seizure prediction." Clinical neurophysiology 120.11 (2009): 1927-1940.
  Note that this network can only be used with num_inputs=60 in config file.
  '''
  print(x)
  x=tf.transpose(x,[0,2,3,1])
  print(x)
  conv1 = tf.layers.conv2d(
      inputs=x,
      filters=5,
      strides=1,
      kernel_size=[1,13],
      padding="valid",
      data_format='channels_last',
      activation=tf.nn.relu,
      kernel_regularizer= tf.contrib.layers.l1_regularizer(scale=0.001))
  print(conv1)
  pool1 = tf.layers.average_pooling2d(
      conv1,
      pool_size=[1,2],
      strides=[1,2],
      padding="valid",
      data_format='channels_last')
  print(pool1)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=5,
      strides=1,
      kernel_size=[cfg.N_features, 9],
      padding="valid",
      data_format='channels_last',
      activation=tf.nn.relu,
      kernel_regularizer= tf.contrib.layers.l1_regularizer(scale=0.001))
  print(conv2)
  pool2 = tf.layers.average_pooling2d(
      conv2,
      pool_size=[1,2],
      strides=[1,2],
      padding="valid",
      data_format='channels_last')
  print(pool2)
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=3,
      strides=1,
      kernel_size=[1, 8],
      padding="valid",
      activation=tf.nn.relu,
      data_format='channels_last',
      kernel_regularizer= tf.contrib.layers.l1_regularizer(scale=0.001))
  print(conv3)
  output = tf.layers.dense(conv3, cfg.num_classes, activation=None)
  print(output)
  output = tf.reshape(output, [ tf.shape(output)[0] , 2 ])
  print(output)
  return output
