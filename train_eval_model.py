import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from data_processing.data_loader import loadData
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import metrics #confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, balanced_accuracy_score, accuracy_score, roc_curve

from helpers.plots import plot_cv_indices, plot_roc_auc, plot_cv_splits, plot_confusion_matrix
from models.FC import FCNetwork
from models.RNN import RNNNetwork
from models.TCN import TCNNetwork


# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Just disables the warning ("Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"), doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--patient", help="Patient number")
parser.add_argument("--model", help="Which neural network to use")
parser.add_argument("--data_path", help="Path to load data from binary files")
parser.add_argument("--group_segments_form_input", help="bool: Group segments to form inputs.")
parser.add_argument("--preictal_duration", help="Load data from specified preictal duration (e.g. 1800s or 30min)")
parser.add_argument("--discard_data_duration", help="int: which discard criteria data to use (e.g. 250min of discard data = 4 hours")
parser.add_argument("--n_segments_form_input", help="int: How many segments to use to form one input. This works iff group_segments_form_input==True.")
args = parser.parse_args()


# Define some hyper-parameters
DATA_DIRECTORY = args.data_path + '/chb{:02d}'.format(int(args.patient)) + '/preictal-' + str(int(int(args.preictal_duration) / 60))  + '-discard-' + str(int(int(args.discard_data_duration) / 60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']
GROUP_SEGMENTS_FORM_INPUT = eval(args.group_segments_form_input)  # if True, use N segments and group together to form one input
N_SEGMENTS_FORM_INPUT = int(args.n_segments_form_input)  # number of segments to form an input
PREICTAL_DURATION = int(args.preictal_duration)
PATIENT = int(args.patient)
MODEL = args.model
BUFFER_SIZE = 500000
BATCH_SIZE = 16
CV = 5

print("---------------------------------------------------------------------------------------------------------------------------------------------------")
print("Model: ", MODEL)
print("Patient: ", PATIENT)
print("Preictal duration: ", PREICTAL_DURATION)
if GROUP_SEGMENTS_FORM_INPUT:
    print('Nr of segments to stack: ', N_SEGMENTS_FORM_INPUT)


print("Loading the data")
X, Y = loadData(DATA_DIRECTORY, FILE_NAMES, GROUP_SEGMENTS_FORM_INPUT, N_SEGMENTS_FORM_INPUT)

print("X shape: ", np.shape(X))
print("Y shape: ", np.shape(Y))


# Calculate input dimensionality
N_FEATURES = np.shape(X)[1]  # data features/dimensionality
INPUT_DIM = N_FEATURES * N_SEGMENTS_FORM_INPUT if GROUP_SEGMENTS_FORM_INPUT == True else N_FEATURES

# if use of RNN and group segments to form timesteps inputs, reshape the X to [timesteps,features] => timesteps = N_SEGMENTS_FORM_INPUT
if(GROUP_SEGMENTS_FORM_INPUT == True and MODEL == "RNN"):
    FEATURES_ORIGINAL_SIZE = int(N_FEATURES / N_SEGMENTS_FORM_INPUT)
    X = X.reshape([-1, N_SEGMENTS_FORM_INPUT, FEATURES_ORIGINAL_SIZE])

print("X shape after reshaped: ", np.shape(X))

# print("Splitting the data")
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y) # , stratify=y
#
#
# # Samples per class
# unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
# unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
# print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
# print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))



# Load parameters' from JSON
PARAMETERES_FILE_PATH = './CV_results/' + MODEL + '/' + 'chb{:02d}'.format(PATIENT) + '/preictal_' + str(int(PREICTAL_DURATION / 60))  + '_discard_' + str(int(int(args.discard_data_duration) / 60)) + '_best_params.json'
with open(PARAMETERES_FILE_PATH, 'r') as JSON_FILE:
    PARAMETERS = json.load(JSON_FILE)
print("Hyper-parameters: ", PARAMETERS)


# select neural network
print('Creating the model...')
if (MODEL == "FC"):
    model = FCNetwork.build_model(
        input_dim=INPUT_DIM,
        units1=PARAMETERS['units1'],
        units2=PARAMETERS['units2'],
        dropout1=PARAMETERS['dropout1'],
        dropout2=PARAMETERS['dropout2'],
        learning_rate=PARAMETERS['learning_rate'],
        multi_layer=PARAMETERS['multi_layer'],
        l2_1=PARAMETERS['l2_1'],
        l2_2=PARAMETERS['l2_2'],
        kernel_init=PARAMETERS['kernel_init']
    )
    print("Model architecture: ", model.summary())
elif (MODEL == "TCN"):
    model = TCNNetwork.build_model()
elif (MODEL == "RNN"):
    model = RNNNetwork.build_model(
        input_dim=FEATURES_ORIGINAL_SIZE,
        timesteps=N_SEGMENTS_FORM_INPUT,
        units1=PARAMETERS['units1'],
        units2=PARAMETERS['units2'],
        units3=PARAMETERS['units3'],
        dropout1=PARAMETERS['dropout1'],
        dropout2=PARAMETERS['dropout2'],
        dropout3=PARAMETERS['dropout3'],
        learning_rate=PARAMETERS['learning_rate'],
        multi_layer=PARAMETERS['multi_layer'],
        l2_1=PARAMETERS['l2_1'],
        l2_2=PARAMETERS['l2_2'],
        l2_3=PARAMETERS['l2_3'],
        kernel_init=PARAMETERS['kernel_init']
    )
    print("Model architecture: ", model.summary())

print("Started training...")
start = time.time()

# Because of imbalanced data, calculate class weights
class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
print("Class weights: ", class_weights)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1)
]



'''
#
#
# Plot the cv
#
#
'''
# define 10-fold cross validation test harness
cv_skfold = StratifiedKFold(n_splits=CV, shuffle=True, random_state=seed)
cv_kfold = KFold(n_splits=CV, shuffle=True, random_state=seed)

# Plot cv fold data splits - we use StratifiedKFold, but compare with KFold using the plot
plot_cv_splits(cv_skfold, X, Y, PREICTAL_DURATION, PATIENT, 'skfold_data_plot')
plot_cv_splits(cv_kfold, X, Y, PREICTAL_DURATION, PATIENT, 'kfold_data_plot')


# hold tprs, aucs and the mean to plot the ROC AUC
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


final_acc = []
final_balanced_acc = []
final_precision = []
final_recall = []
final_f1 = []
final_auc = []
final_fpr = []
final_tpr = []
final_tnr = []

tps = []
tns = []
fps = []
fns = []

plt.figure(figsize=(8,6))
kth = 0
for train, test in cv_skfold.split(X, Y):
    kth += 1
    print("=================== FOLD: ", kth ,"======================================================================================================================================")

    # Samples per class
    unique_y_train, counts_y_train = np.unique(Y[train], return_counts=True)
    unique_y_test, counts_y_test = np.unique(Y[test], return_counts=True)
    print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
    print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))


    # Load data to tf.data.Dataset, shuffle and create batches
    train_set = tf.data.Dataset.from_tensor_slices((X[train], Y[train])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE,                                                                      drop_remainder=False)
    test_set = tf.data.Dataset.from_tensor_slices((X[test], Y[test])).batch(BATCH_SIZE, drop_remainder=False)

    # fit the model
    model.fit(train_set, epochs=30, validation_data=test_set, class_weight=class_weights, callbacks=callbacks)

    # evaluate the model
    loss, acc = model.evaluate(test_set)
    print("Eval Loss: ", loss, "\nEval Accuracy: ", acc)
    print('\n\n')

    '''
    ###
    ### Prediction - metrics calculations
    ###
    '''
    # predict probabilities for test set
    y_pred_probs = model.predict(X[test]).ravel() # ravel() reduce to 1d array

    # predict crisp classes for test set
    y_pred_classes = model.predict_classes(X[test]).ravel() # ravel() reduce to 1d array

    # Calculate the metrics
    # accuracy: (tp + tn) / (p + n)
    accuracy = metrics.accuracy_score(Y[test], y_pred_classes)
    final_acc.append(accuracy)
    print('Accuracy: ', accuracy)

    # balanced accuracy: n_samples / (n_classes * np.bincount(y))
    balanced_accuracy = metrics.balanced_accuracy_score(Y[test], y_pred_classes)
    final_balanced_acc.append(balanced_accuracy)
    print('Balanced accuracy: ', balanced_accuracy)

    # precision tp / (tp + fp)
    precision = metrics.precision_score(Y[test], y_pred_classes)
    final_precision.append(precision)
    print('Precision: ', precision)

    # recall: tp / (tp + fn)
    recall = metrics.recall_score(Y[test], y_pred_classes)
    final_recall.append(recall)
    print('Recall: ', recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(Y[test], y_pred_classes)
    final_f1.append(f1)
    print('F1 score: ', f1)

    # Additional metrics
    # ROC AUC
    auc = metrics.roc_auc_score(Y[test], y_pred_probs)
    final_auc.append(auc)
    print('ROC AUC: ', auc)

    # Confusion matrix
    # true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
    # true negatives (TN): We predicted no, and they don't have the disease.
    # false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
    # false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
    print("Confusion matrix: \n", metrics.confusion_matrix(Y[test], y_pred_classes))
    tn, fp, fn, tp = metrics.confusion_matrix(Y[test], y_pred_classes).ravel()
    tps.append(tp)
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)
    print({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})

    fpr = fp/(fp+tn)
    final_fpr.append(fpr)
    print('FPR: ', fpr)

    tpr = tp/(tp+fn)
    final_tpr.append(tpr)
    print('TPR (sensitivity): ', tpr)

    tnr = tn/(fp+tn)
    final_tnr.append(tnr)
    print('TNR (specificity): ', tnr)


    # Plot ROC AUC
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(Y[test], y_pred_probs)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (kth, roc_auc))
    print("=========================================================================================================================================================")

plot_roc_auc(tprs, mean_fpr, aucs, MODEL, PATIENT, plt)

cm_constructed = np.array([int(np.sum(tns) / CV), int(np.sum(fps) / CV), int(np.sum(fns) / CV), int(np.sum(tps) / CV)]).reshape((2, 2))
plot_confusion_matrix(cm_constructed, MODEL, PATIENT, plt, classes=[0,1], title='Confusion matrix')

print('Report the results:')
print('Final confusion matrix, without normalization: \n', cm_constructed)
print('Final acc', np.mean(final_acc))
print('Final balanced_acc', np.mean(final_balanced_acc))
print('Final precision', np.mean(final_precision))
print('Final recall', np.mean(final_recall))
print('Final f1', np.mean(final_f1))
print('Final auc', np.mean(final_auc))
print('Final fpr', np.mean(final_fpr))
print('Final tpr', np.mean(final_tpr))
print('Final tnr', np.mean(final_tnr))

print("---------------------------------------------------------------------------------------------------------------------------------------------------")