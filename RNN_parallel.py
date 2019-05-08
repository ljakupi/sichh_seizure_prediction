from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pyedflib
import glob
import re
import numpy as np
import random
import EEG

flags = tf.flags
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
FLAGS = flags.FLAGS                   

#data parameters
sampling=256 #Hz
duration=5 # duration of samples to train

# Training Parameters
learning_rate = 0.001
training_steps = 100
batch_size = 10
display_step = 1

# Network Parameters
num_input = 20 # data input (# signals)
# ~ timesteps = 1155 # timesteps
timesteps = 815 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 2 # Two classes: Inter and pre

data_division={}
patient_data = EEG.Patient_data(data_division,FLAGS.data_path)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

def RNN(x):
  # GPU version
  # ~ lstm_cuda = tf.contrib.cudnn_rnn.CudnnLSTM(1,num_hidden)
  # ~ outputs, _ = lstm_cuda(x)
  lstm_cell = tf.contrib.rnn.LSTMBlockCell(num_hidden, forget_bias=1.0)
  # ~ lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden, forget_bias=1.0)
  outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=x, dtype=tf.float32)
  # we take only the output at the last time
  # ~ fc_layer = tf.layers.dense(outputs[:,-1,:], 30, activation=tf.nn.relu)
  # we take all outputs
  outputs_1D = tf.reshape(outputs,[-1,num_hidden*timesteps])
  fc_layer = tf.layers.dense(outputs_1D, 30, activation=tf.nn.relu)
  output_layer = tf.layers.dense(fc_layer, num_classes, activation=None)
  return output_layer

logits = RNN(X)
# ~ prediction = tf.nn.sigmoid(logits)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
# ~ loss_op = tf.losses.sigmoid_cross_entropy(multi_class_labels = Y, logits = logits)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Start trainingrun 
with tf.Session() as sess:
  
  # Run the initializer
  sess.run(init)

  for step in range(1, training_steps+1):
    batch_x, batch_y = patient_data.train_next_batch(batch_size, num_input)
    # Reshape data ? Check this
    batch_x = batch_x.reshape((batch_size, timesteps, num_input))
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
    if step % display_step == 0 or step == 1:
      # Calculate batch loss and accuracy
      loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
      print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
  save_path = saver.save(sess, "./ckpt/18_100.ckpt")
  print("Model saved in path: %s" % save_path)
  print("Optimization Finished!")

    # Calculate accuracy 
  test_len = 10
  test_data, test_label = patient_data.get_test_batch(test_len, num_input)
  test_data = test_data.reshape((test_len, timesteps, num_input))
  acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
  print("Testing Accuracy:" + "{:.3f}".format(acc))
