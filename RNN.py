from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib


def RNN(x, cfg):

    # GPU version

    # input_rs shape after transpose [num_batch, num_inputs, num_features]
    # inputs_rs = tf.transpose(x, perm=[0, 2, 1])
    inputs_rs = x

    # create a RNN cell composed sequentially of a number of RNNCells/LSTMCell
    stacked_rnn = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.LSTMCell(num_units=cfg.num_hidden, forget_bias=1.0) for _ in range(cfg.num_layers)
    ])

    # add the operations to the graph that simulate the recurrent network over the time steps of the input
    # 'outputs' is a tensor of shape [batch_size, max_time, num_hidden]
    outputs, _ = tf.nn.dynamic_rnn(cell=stacked_rnn, inputs=inputs_rs, dtype=tf.float32, parallel_iterations=10)


    # Take only last LSTM output
    # fc_layer = tf.layers.dense(outputs[:,-1,:], units=30, activation=tf.nn.relu)

    # Take all LSTM outputs
    outputs_1D = tf.reshape(outputs, [-1, cfg.num_hidden * cfg.num_inputs])
    fc_layer = tf.layers.dense(outputs_1D, units=30, activation=tf.nn.relu)

    output_layer = tf.layers.dense(fc_layer, cfg.num_classes, activation=None)

    return output_layer
