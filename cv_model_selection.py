import os
import json
import argparse
import numpy as np
import time, datetime
import tensorflow as tf

from sklearn.utils import class_weight
from data_processing.data_loader import loadData
from sklearn.model_selection import train_test_split, StratifiedKFold

from models.FC import FCNetwork
from models.RNN import RNNNetwork
from models.TCN import TCNNetwork

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Just disables the warning ("Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"), doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# Using scikit-optimize as hyperparameter optimization tool
# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


parser = argparse.ArgumentParser()
parser.add_argument("--patient", help="Patient number")
parser.add_argument("--model", help="Which neural network to use")
parser.add_argument("--CV", help="How many folds to use for cross validation")
parser.add_argument("--data_path", help="Path to load data from binary files")
parser.add_argument("--group_segments_form_input", help="bool: Group segments to form inputs.")
parser.add_argument("--preictal_duration", help="Load data from specified preictal duration (e.g. 1800s or 30min)")
parser.add_argument("--discard_data_duration", help="int: which discard criteria data to use (e.g. 250min of discard data = 4 hours")
parser.add_argument("--n_segments_form_input", help="int: How many segments to use to form one input. This works iff group_segments_form_input==True.")
parser.add_argument("--segmentation_duration", help="str: Duration used for data segmentation during the feature calculations (e.g. 30 sec or 5 sec).")
args = parser.parse_args()


# Define some hyper-parameters
DATA_DIRECTORY = args.data_path + '/sec-' + args.segmentation_duration + '/chb{:02d}'.format(int(args.patient)) + '/preictal-' + str(int(int(args.preictal_duration) / 60))  + '-discard-' + str(int(int(args.discard_data_duration) / 60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']
GROUP_SEGMENTS_FORM_INPUT = eval(args.group_segments_form_input)  # if True, use N segments and group together to form one input
N_SEGMENTS_FORM_INPUT = int(args.n_segments_form_input)  # number of segments to form an input
PREICTAL_DURATION = int(args.preictal_duration)
PATIENT = int(args.patient)
MODEL = args.model
CV = int(args.CV)
BUFFER_SIZE = 500000
EPOCHS = 30

print("CV: ", CV)
print("Epochs: ", EPOCHS)
print("Model: ", MODEL)
print("Patient: ", PATIENT)


print("Loading the data")
X, y = loadData(DATA_DIRECTORY, FILE_NAMES, GROUP_SEGMENTS_FORM_INPUT, N_SEGMENTS_FORM_INPUT)


# Calculate input dimensionality
N_FEATURES = np.shape(X)[1]  # data features/dimensionality
INPUT_DIM = N_FEATURES * N_SEGMENTS_FORM_INPUT if GROUP_SEGMENTS_FORM_INPUT == True else N_FEATURES

# if use of RNN and group segments to form timesteps inputs, reshape the X to [timesteps,features]. timesteps = N_SEGMENTS_FORM_INPUT
if(GROUP_SEGMENTS_FORM_INPUT == True and (MODEL == "RNN" or MODEL == "TCN")):
    FEATURES_ORIGINAL_SIZE = int(N_FEATURES / N_SEGMENTS_FORM_INPUT)
    X = X.reshape([-1, N_SEGMENTS_FORM_INPUT, FEATURES_ORIGINAL_SIZE])

print("X shape", np.shape(X))

print("Splitting the data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

# Samples per class
unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))


# select neural network
# wrap the TF model with KerasClassifier()
print('Creating the model...')
if (MODEL == "FC"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build_model, input_dim=INPUT_DIM, epochs=EPOCHS, verbose=0)
elif (MODEL == "TCN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=TCNNetwork.build_model, input_dim=FEATURES_ORIGINAL_SIZE, timesteps=N_SEGMENTS_FORM_INPUT, verbose=0)
elif (MODEL == "RNN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=RNNNetwork.build_model, input_dim=FEATURES_ORIGINAL_SIZE, timesteps=N_SEGMENTS_FORM_INPUT, epochs=EPOCHS, verbose=0)


# prepare the parameters' grid
params_dict = {
    'units1': Integer(low=16, high=128, name='units1'),
    'units2': Integer(low=16, high=128, name='units2'),
    'dropout1': Real(low=0.01, high=0.9, prior='log-uniform', name='dropout1'),
    'dropout2': Real(low=0.01, high=0.9, prior='log-uniform', name='dropout2'),
    'learning_rate': Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
    'l2_1': Real(low=0.01, high=0.9, prior='log-uniform', name='l2_1'),
    'l2_2': Real(low=0.01, high=0.9, prior='log-uniform', name='l2_2'),
    'kernel_init': Categorical(categories=['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'],
                               name='kernel_init'),
    'multi_layer': Categorical(categories=[True, False], name='multi_layer'),
    'sgd_opt': Categorical(categories=[True, False], name='sgd_opt'),
    'moment': Real(low=0.01, high=0.9, prior='log-uniform', name='moment'),
}

if MODEL == "RNN" or MODEL == "FC":
    additional_params_dict = {
        'units3': Integer(low=16, high=128, name='units3'),
        'dropout3': Real(low=0.01, high=0.9, prior='log-uniform', name='dropout3'),
        'l2_3': Real(low=0.01, high=0.9, prior='log-uniform', name='l2_3')
    }

    # add additional params
    params_dict.update(additional_params_dict)


print("Started training...")
start = time.time()

# Because of imbalanced data, calculate class weights
class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
print("Class weights: ", class_weights)

#define partitioning
SKF = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
print('SKF splits: ', SKF.get_n_splits(X_train, y_train))

# run the grid search and cross-validation
search_params = BayesSearchCV(estimator=model, cv=SKF, n_iter=10, fit_params={'class_weight': class_weights}, search_spaces=params_dict, scoring='accuracy')

# callback handler to stop when optimal loss found
# stop the exploration if we reach score >= 0.99
# https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html
def on_step(optim_result):
    score = search_params.best_score_
    print("Best score: %s" % score)
    if score >= 0.99:
        print('Interrupting!')
        return True

search_result = search_params.fit(X_train, y_train, callback=on_step)

print("Best results: \n", search_result.best_params_, '\n')

# write the best model params to a JSON file
# helper function to cast np.int64 integers to int because JSON does not accept np.int64
def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

PARENT_DIR = "./CV_results/" + MODEL + '/sec-' + args.segmentation_duration + '/' + 'chb{:02d}'.format(PATIENT)
os.makedirs(PARENT_DIR, exist_ok=True)  # create any parent directory that does not exist
best_model_file = open(PARENT_DIR + '/preictal_' + str(int(PREICTAL_DURATION / 60))  + '_discard_' + str(int(int(args.discard_data_duration) / 60)) + "_best_params.json", "w")
best_model_file.write(json.dumps(search_result.best_params_, default=default))
best_model_file.close()

end = time.time()
res = (end - start)
print("Finished training...: ", str(datetime.timedelta(seconds=res)))


print("Val. score: %s" % search_result.best_score_)
print("Test score: %s" % search_result.score(X_test, y_test))

print('Best params: \n', search_result.best_params_)