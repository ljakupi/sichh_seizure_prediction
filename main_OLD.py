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

# tf Graph input
print('N features', cfg.N_features)
print('num inputs', cfg.num_inputs)
print('num classes', cfg.num_classes)
# X = tf.placeholder("float", [1, cfg.N_features, cfg.num_inputs])
X = tf.placeholder("float", [None, cfg.N_features, cfg.num_inputs])
# Y = tf.placeholder("float", [cfg.num_classes])
Y = tf.placeholder("float", [None, cfg.num_classes])

if (cfg.NN == "TCN"):
    logits = TCN.TCN(X, cfg)
elif (cfg.NN == "FC"):
    logits = FC.FC(X, cfg)
elif (cfg.NN == "RNN"):
    logits = RNN.RNN(X, cfg)
elif (cfg.NN == "CNN"):
    logits = CNN.CNN(X, cfg)

# ~ prediction = tf.nn.sigmoid(logits)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
l1_loss = tf.losses.get_regularization_loss()
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# ~ loss_op = tf.losses.log_loss(labels=Y, predictions=prediction, loss_collection=None)
loss_op += l1_loss
# ~ loss_op = tf.losses.sigmoid_cross_entropy(multi_class_labels = Y, logits = logits)

# ~ optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(loss_op)
optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate, momentum=0.9).minimize(loss_op)
# ~ optimizer = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss_op)
# ~ optimizer = tf.train.AdagradOptimizer(cfg.learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
save_file = "./ckpt/" + cfg.NN + ".ckpt"


# Start training run
# ~ with tf.Session() as sess:
def run():
    train_loss_history = []
    train_acc_history = []

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)) as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, cfg.training_steps + 1): # epochs

            batch_x, batch_y = patient_data.train_next_batch(cfg.batch_size)

            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            if step % cfg.display_step == 0 or step == 1:

                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                # Append training accuracy and loss per epoch for plotting
                train_loss_history.append(loss)
                train_acc_history.append(acc)

                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                # test_data, test_label = patient_data.get_test_batch(cfg.test_len)
                # test_data, test_label = patient_data.get_test_batch(cfg.batch_size)
                # acc, pred = sess.run([accuracy, prediction], feed_dict={X: test_data, Y: test_label})

                # # Append validation accuracy and loss per epoch for plotting
                # val_acc_history.append(acc)
                # val_loss_history.append(val_loss)
                # print("Validation Accuracy:" + "{:.3f}".format(acc))

        save_path = saver.save(sess, save_file)
        print("Model saved in path: %s" % save_path)
        print("Optimization Finished!")

        # Test the network
        test_data, test_label = patient_data.get_test_batch(cfg.test_len)
        acc, pred = sess.run([accuracy, prediction], feed_dict={X: test_data, Y: test_label})
        print("Testing Accuracy:" + "{:.3f}".format(acc))

        # Print classification report
        print('Classification report: \n', classification_report(np.argmax(test_label, axis=1), np.argmax(pred, axis=1)))

        # Plot confusion matrix
        confusionMatrixPlot(np.argmax(test_label, axis=1), np.argmax(pred, axis=1))

        # Plot loss and accuracy
        plotAccuracyAndLoss(train_loss_history, 'loss')
        plotAccuracyAndLoss(train_acc_history, 'accuracy')

def confusionMatrixPlot(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    scores = [0, 1]
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=scores, yticklabels=scores)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig('./images/chb02-confusion-matrix.png')

def plotAccuracyAndLoss(metricData, lossOrAcc):
    fig = plt.figure(figsize=(15, 8))
    font = {'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 12, }
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(metricData, 'r', label='Training %(lossOrAcc)s' % {'lossOrAcc': lossOrAcc})
    ax1.set_ylabel('%(lossOrAcc)s' % {'lossOrAcc': lossOrAcc}, fontdict=font)
    ax1.set_title("%(lossOrAcc)s plot" % {'lossOrAcc': lossOrAcc}, fontdict=font)
    ax1.set_xlabel('Epoch', fontdict=font)
    plt.savefig('./images/chb02-training-%(lossOrAcc)s.png' % {'lossOrAcc': lossOrAcc})

run()
print('Segment type train: ', cfg.segments_type_train, '\nSegment type test: ', cfg.segments_type_test)
