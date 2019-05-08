from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib
import pyedflib
import cProfile
import glob
import re
import numpy as np
import random
import EEG
import config
import FC
import TCN
import RNN
import CNN

####### Additional ######
import os
from itertools import zip_longest
from tensorflow import contrib
tfe = contrib.eager
import math
import time
#########################

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


flags = tf.flags
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("NN", None,
                    "Type of neural network.")
flags.DEFINE_string("patient", None,
                    "Patient number")
FLAGS = flags.FLAGS
cfg = config.Config(data_path=FLAGS.data_path, NN=FLAGS.NN, patient=int(FLAGS.patient))
patient_data = EEG.Patient_data(cfg)
cfg.N_features = np.shape(patient_data.segments)[1]
tf.random.set_random_seed(1)


###################### Create graph ######################
# Placeholders for inputs (x), outputs(y)
with tf.variable_scope('Input'):
    X = tf.placeholder(tf.float32, shape=[None, cfg.num_inputs, cfg.N_features], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, cfg.num_classes], name='Y')

# Select neural network
if (cfg.NN == "TCN"):
    output_logits = TCN.TCN(X, cfg)
elif (cfg.NN == "FC"):
    output_logits = FC.FC(X, cfg)
elif (cfg.NN == "RNN"):
    output_logits = RNN.RNN(X, cfg)
elif (cfg.NN == "CNN"):
    output_logits = CNN.CNN(X, cfg)

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        l2_loss = tf.losses.get_regularization_loss()

        # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output_logits), name='loss')

        classes_weights = tf.constant([0.1, 9.0])
        loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y, logits=output_logits, pos_weight=classes_weights), name='loss')

        loss_op += l2_loss

    with tf.variable_scope('Optimizer'):
        # optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate, name='adam-optimizer').minimize(loss_op)
        optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate, momentum=0.9, name='momentum-optimizer').minimize(loss_op)

    with tf.variable_scope('Accuracy'):
        predicted = tf.argmax(output_logits, 1, name='predicted-labels')
        correct = tf.argmax(Y, 1, name='correct-labels')
        correct_predicted_labels = tf.equal(predicted, correct, name='correct-predicted-labels')
        accuracy = tf.reduce_mean(tf.cast(correct_predicted_labels, tf.float32), name='accuracy')

# Define directories to store summaries
dirname = "run-" + cfg.NN + "-" + time.strftime("%H%M%S") # e.g. run-RNN-160611
train_summary_dir = os.path.join('./', "summaries", "train", dirname) # set the directory to save summaries
val_summary_dir = os.path.join('./', "summaries", "val", dirname) # set the directory to save summaries
test_summary_dir = os.path.join('./', "summaries", "test", dirname) # set the directory to save summaries

# Save the model
saver = tf.train.Saver()
save_file = "./ckpt/" + cfg.NN + ".ckpt"

# Initializing the variables
init = tf.global_variables_initializer()

# Training
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)) as sess:

    # Run the initializer
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    # Prepare training the data
    X_training, Y_training = patient_data.prepare_X_and_y_data(training=True, testing=False)

    # Prepare val the data
    X_val, Y_val = patient_data.prepare_X_and_y_data(training=False, testing=False)
    # X_val, Y_val = patient_data.prepare_X_and_y_data(training=False, testing=True) # use full test set for

    print("X_training: ", np.shape(X_training))
    print("Y_training: ", np.shape(Y_training))
    print("X_val: ", np.shape(X_val))
    print("Y_val: ", np.shape(Y_val))

    # Used to determine when to stop the training early
    valid_loss_summary = []
    stop_early_step = 0

    # Training metrics summaries for reporting and ploting
    train_epoch_loss = []
    train_epoch_acc = []

    # Validation metrics summaries for reporting and ploting
    val_epoch_loss = []
    val_epoch_acc = []

    # Training loop
    for epoch in range(1, cfg.training_steps + 1):

        # Get training batches
        X_train_batches = patient_data.X_training_batches_generator(X_training, cfg.batch_size)
        Y_train_batches = patient_data.X_training_batches_generator(Y_training, cfg.batch_size)

        # Get training batches
        X_val_batches = patient_data.X_training_batches_generator(X_val, cfg.batch_size)
        Y_val_batches = patient_data.X_training_batches_generator(Y_val, cfg.batch_size)

        for batch_x_train, batch_y_train in zip_longest(X_train_batches, Y_train_batches):

            # Save batches in lists (memory) because generator generates data on fly, therefore, after one batch, the data are lost!
            batch_x_list_train = list(batch_x_train)
            batch_y_list_train = list(batch_y_train)

            # Define batch dict to feed
            feed_dict_batch = {
                X: batch_x_list_train,
                Y: batch_y_list_train
            }

            # Run the optimizer
            _, loss_batch, acc_batch = sess.run([
                optimizer,
                loss_op,
                accuracy
            ], feed_dict_batch)

            # Record the loss and accuracy of each training batch
            train_epoch_loss.append(loss_batch)
            train_epoch_acc.append(acc_batch)


        # Average the training loss and accuracy of each epoch
        avg_train_loss = np.mean(train_epoch_loss)
        avg_train_acc = np.mean(train_epoch_acc)

        # Generate training summaries per epoch for TensorBoard plotting
        summary_loss = tf.Summary(value=[tf.Summary.Value(tag="Training Loss", simple_value=avg_train_loss)])
        summary_accu = tf.Summary(value=[tf.Summary.Value(tag="Training Accuracy", simple_value=avg_train_acc)])
        train_summary_writer.add_summary(summary_loss, epoch)
        train_summary_writer.add_summary(summary_accu, epoch)

        # ##################################### Validation #####################################
        for batch_x_val, batch_y_val in zip_longest(X_val_batches, Y_val_batches):

            # Save batches in lists (memory) because generator generates data on fly, therefore, after one epoch, the data are lost!
            batch_x_list_val = list(batch_x_val)
            batch_y_list_val = list(batch_y_val)

            # Define batch dict to feed
            feed_dict_batch = {
                X: batch_x_list_val,
                Y: batch_y_list_val
            }

            # Run the network
            val_loss_batch, val_acc_batch = sess.run([
                loss_op,
                accuracy
            ], feed_dict_batch)

            # Record the loss and accuracy of each training batch
            val_epoch_loss.append(val_loss_batch)
            val_epoch_acc.append(val_acc_batch)

        # Average the training loss and accuracy of each epoch
        avg_val_loss = np.mean(val_epoch_loss)
        avg_val_acc = np.mean(val_epoch_acc)
        best_current_loss = min(valid_loss_summary, default=0) # get the best current loss - before adding the new best one
        valid_loss_summary.append(avg_val_loss)

        # Generate validation summaries per epoch for TensorBoard plotting
        summary_val_loss = tf.Summary(value=[tf.Summary.Value(tag="Val Loss", simple_value=avg_val_loss)])
        summary_val_accu = tf.Summary(value=[tf.Summary.Value(tag="Val Accuracy", simple_value=avg_val_acc)])
        val_summary_writer.add_summary(summary_val_loss, epoch)
        val_summary_writer.add_summary(summary_val_accu, epoch)
        # ##################################### Validation #####################################

        # Print the progress of each epoch
        print("Epoch: ", epoch,
              " | Train Loss: {:.3f}".format(avg_train_loss),
              " | Train Acc: {:.3f}".format(avg_train_acc),
              " | Valid Loss: {:.3f}".format(avg_val_loss),
              " | Valid Acc: {:.3f}".format(avg_val_acc))

        # Stop training if the validation loss does not decrease after 3 epochs
        # Since loss value is of type float, compare the numbers up to 5 decimals after the dot
        if round(avg_val_loss,5) >= round(best_current_loss,5):
            stop_early_step += 1
            if stop_early_step == 3:
                print("No Improvement. Stop training!")
                break
        # Reset stop_early if the validation loss finds a new low
        # Save a checkpoint of the model
        else:
            stop_early_step = 0
            save_path = saver.save(sess, save_file)

    ################################################ TESTING ############################################################

    print("Testing .... ")

    # Prepare testing the data
    X_test, Y_test = patient_data.prepare_X_and_y_data(training=False, testing=True)

    print("X_test: ", np.shape(X_test))
    print("Y_test: ", np.shape(Y_test))

    # Test metrics summaries for reporting and ploting
    test_epoch_loss = []
    test_epoch_acc = []

    # Predicted labels
    predicted_labels_all = []

    # Get test batches
    # X_test_batches = patient_data.X_training_batches_generator(X_test, cfg.batch_size)
    # Y_test_batches = patient_data.X_training_batches_generator(Y_test, cfg.batch_size)

    # Define batch dict to feed
    feed_dict_batch = {
        X: X_test,
        Y: Y_test
    }

    # Run the optimizer
    predicted_ys, loss_test, acc_test = sess.run([
        predicted,
        loss_op,
        accuracy
    ], feed_dict_batch)

    # Evaluate accuracy
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test loss:', loss_test)
    print('Test accuracy:', acc_test)

    print('\nConfusion matrix:')
    true_labels = tf.argmax(Y_test,1)
    cm = tf.confusion_matrix(true_labels, predicted_ys)
    print(sess.run(cm))


    #
    #
    # batch_step = 0
    #
    # for batch_x_test, batch_y_test in zip_longest(X_test_batches, Y_test_batches):
    #     # Save batches in lists (memory) because generator generates data on fly, therefore, after one batch, the data are lost!
    #     batch_x_list_test = list(batch_x_test)
    #     batch_y_list_test = list(batch_y_test)
    #
    #     # Define batch dict to feed
    #     feed_dict_batch = {
    #         X: batch_x_list_test,
    #         Y: batch_y_list_test
    #     }
    #
    #     # Run the optimizer
    #     predicted_ys, loss_batch, acc_batch = sess.run([
    #         predicted,
    #         loss_op,
    #         accuracy
    #     ], feed_dict_batch)
    #
    #     # Collect predicted labels for all batches
    #     predicted_labels_all.append(predicted_ys)
    #
    #     # Record the loss and accuracy of each training batch
    #     test_epoch_loss.append(loss_batch)
    #     test_epoch_acc.append(acc_batch)
    #
    #     batch_step += 1
    #
    # # Average the training loss and accuracy of each epoch
    # avg_test_loss = np.mean(test_epoch_loss)
    # avg_test_acc = np.mean(test_epoch_acc)
    #
    # # Generate training summaries per epoch for TensorBoard plotting
    # summary_test_loss = tf.Summary(value=[tf.Summary.Value(tag="Test Loss", simple_value=avg_test_loss)])
    # summary_test_accu = tf.Summary(value=[tf.Summary.Value(tag="Test Accuracy", simple_value=avg_test_acc)])
    # test_summary_writer.add_summary(summary_test_loss, batch_step)
    # test_summary_writer.add_summary(summary_test_accu, batch_step)
    #
    # # Print the progress of each epoch
    # print("Test Loss: {:.3f}".format(avg_test_loss), " | Test Acc: {:.3f}".format(avg_test_acc))
    #
    # print("predicted_labels_all: ", np.shape(predicted_labels_all))

    ############################################## END TESTING ##########################################################

print('\n Finished.')
# print('Segment type train: ', cfg.segments_type_train, '\nSegment type test: ', cfg.segments_type_test)