import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from models.FC import FCNetwork
from models.CNN import CNNNetwork
from models.RNN import RNNNetwork
from models.TCN import TCNNetwork
from data_processing.data_loader import loadData
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score, roc_auc_score, balanced_accuracy_score

# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="string: path to load data from binary files")
parser.add_argument("--patient", help="int: patient number")
parser.add_argument("--model", help="string: which model")
parser.add_argument("--preictal_duration",
                    help="int: which preictal prepared data to use (e.g. 1800s or 30min)")
parser.add_argument("--group_segments_form_input", help="bool: group segments to form inputs")
parser.add_argument("--n_segments_form_input",
                    help="int: how many segments to use to form one input. This works iff group_segments_form_input==True.")
args = parser.parse_args()

# Define some hyper-parameters
DATA_DIRECTORY = args.data_path + '/chb{:02d}'.format(int(args.patient)) + '/preictal-' + str(
    int(int(args.preictal_duration) / 60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']
GROUP_SEGMENTS_FORM_INPUT = eval(args.group_segments_form_input)  # if True, group N segments together to form an input
N_SEGMENTS_FORM_INPUT = int(args.n_segments_form_input)
PREICTAL_DURATION = int(args.preictal_duration)
PATIENT = int(args.patient)
MODEL = args.model
BUFFER_SIZE = 500000

print("Loading the data")
X, y = loadData(DATA_DIRECTORY, FILE_NAMES, GROUP_SEGMENTS_FORM_INPUT, N_SEGMENTS_FORM_INPUT)

# Get data features size and load best parameters chosen and stored from cross-validation and model selection part
N_FEATURES = np.shape(X)[1]  # data features/dimensionality
INPUT_DIM = N_FEATURES * N_SEGMENTS_FORM_INPUT if GROUP_SEGMENTS_FORM_INPUT == True else N_FEATURES  # Calculate input dimensionality

# if use of RNN and group segments to form timesteps inputs, reshape the X to [timesteps,features]. timesteps = N_SEGMENTS_FORM_INPUT
if(GROUP_SEGMENTS_FORM_INPUT == True and MODEL == "RNN"):
    FEATURES_ORIGINAL_SIZE = int(N_FEATURES / N_SEGMENTS_FORM_INPUT)
    X = X.reshape([-1, N_SEGMENTS_FORM_INPUT, FEATURES_ORIGINAL_SIZE])

print("X shape", np.shape(X))

# Load parameters' from JSON
PARAMETERES_FILE_PATH = './CV_results/' + MODEL + '/' + 'chb{:02d}'.format(PATIENT) + '/preictal_' + str(
    int(PREICTAL_DURATION / 60)) + '_best_params.json'
with open(PARAMETERES_FILE_PATH, 'r') as JSON_FILE:
    PARAMETERS = json.load(JSON_FILE)
print("Best params: ", PARAMETERS)

# Split training and test data using stratify, which is helpful in cases where data are imbalanced to assure that there is a fair
# split between train and test, both having samples from all classes involved
print("Splitting the data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# Samples per class
unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))

# Load data to tf.data.Dataset, shuffle and create batches
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(PARAMETERS['batch_size'], drop_remainder=False)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(PARAMETERS['batch_size'], drop_remainder=False)

# Select the neural network
print('Creating the model...')
if (MODEL == "FC"):
    model = FCNetwork.build(PARAMETERS['learning_rate'], PARAMETERS['init'], PARAMETERS['dropout'],
                            PARAMETERS['hidden_size'], INPUT_DIM)
elif (MODEL == "TCN"):
    model = TCNNetwork.build(PARAMETERS['learning_rate'], PARAMETERS['init'], PARAMETERS['dropout'],
                             PARAMETERS['hidden_size'], INPUT_DIM)
elif (MODEL == "RNN"):
    model = RNNNetwork.build(PARAMETERS['learning_rate'], PARAMETERS['init'], PARAMETERS['dropout'],
                             PARAMETERS['hidden_size'], FEATURES_ORIGINAL_SIZE, N_SEGMENTS_FORM_INPUT)
elif (MODEL == "CNN"):
    model = CNNNetwork.build(PARAMETERS['learning_rate'], PARAMETERS['init'], PARAMETERS['dropout'],
                             PARAMETERS['hidden_size'], INPUT_DIM)

# Train the network
# Model is evaluated at the end of each training epoch

# Because of imbalanced data, calculate class weights
class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
print("Class weights: ", class_weights)

# Fit the model
print("Training...")
print('train_set ', train_set)
model.fit(train_set, epochs=PARAMETERS['epochs'], validation_data=test_set, class_weight=class_weights)

print("Evaluating...")
# Evaluate the network
model.evaluate(test_set)


'''
Prediction - metrics calculations
'''
# predict probabilities for test set
y_pred_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
y_pred_classes = model.predict_classes(X_test, verbose=0)

# reduce to 1d array
y_pred_probs = y_pred_probs[:, 0]
y_pred_classes = y_pred_classes[:, 0]

# Calculate the metrics
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred_classes)
print('Accuracy: %f' % accuracy)

# balanced accuracy: n_samples / (n_classes * np.bincount(y))
balanced_accuracy = balanced_accuracy_score(y_test, y_pred_classes)
print('Balanced accuracy: %f' % balanced_accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred_classes)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred_classes)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred_classes)
print('F1 score: %f' % f1)

# Additional metrics
# kappa
kappa = cohen_kappa_score(y_test, y_pred_classes)
print('Cohens kappa: %f' % kappa)

# ROC AUC
auc = roc_auc_score(y_test, y_pred_probs)
print('ROC AUC: %f' % auc)

# confusion matrix
# true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
# true negatives (TN): We predicted no, and they don't have the disease.
# false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
# false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
print(confusion_matrix(y_test, y_pred_classes))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()
print({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
