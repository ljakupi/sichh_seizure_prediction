from __future__ import print_function
import tensorflow as tf

'''
  A simple Fully Connected Neural network with two layers, dropout and regularizers
'''

######## Best params ##########

# learning_rate = 0.001
# batch_size = 16
# num_inputs = 10
# num_hidden = 16

# with data:
# len(train_wos):  (2000,)
# len(train_ws):  (717,)
# len(test_wos):  (360,)
# len(test_ws):  (360,)


def FC(x, cfg):
    inputs_1D = tf.reshape(x, [-1, cfg.num_inputs * cfg.N_features])

    first_layer = tf.layers.dense(inputs_1D, cfg.num_hidden, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    first_layer = tf.layers.dropout(first_layer, rate=0.4)
    second_layer = tf.layers.dense(first_layer, cfg.num_classes, activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    return second_layer
