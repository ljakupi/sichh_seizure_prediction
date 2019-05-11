import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from models.CNN import CNNNetwork
from models.FC import FCNetwork
from sklearn.model_selection import GridSearchCV
from data_processing.data_loader import loadData
from models.RNN import RNNNetwork
from models.TCN import TCNNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="Path to load data from binary files")
parser.add_argument("--patient", help="Patient number")
parser.add_argument("--model", help="Which neural network to use")
parser.add_argument("--preictal_duration", help="Load data from specified preictal duration (e.g. 1800s or 30min)")
parser.add_argument("--CV", help="How many folds to use for cross validation")
parser.add_argument("--group_segments_form_input", help="bool: Group segments to form inputs.")
parser.add_argument("--n_segments_form_input",
                    help="int: How many segments to use to form one input. This works iff group_segments_form_input==True.")
args = parser.parse_args()


# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Define some hyper-parameters
DATA_DIRECTORY = args.data_path + '/chb{:02d}'.format(int(args.patient)) + '/preictal-' + str(
    int(int(args.preictal_duration) / 60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']
GROUP_SEGMENTS_FORM_INPUT = eval(args.group_segments_form_input)  # if True, use N segments and group together to form one input
N_SEGMENTS_FORM_INPUT = int(args.n_segments_form_input)  # number of segments to form an input
PREICTAL_DURATION = int(args.preictal_duration)
PATIENT = int(args.patient)
MODEL = args.model
CV = int(args.CV)
BUFFER_SIZE = 500000


print("Loading the data")
X, y = loadData(DATA_DIRECTORY, FILE_NAMES, GROUP_SEGMENTS_FORM_INPUT, N_SEGMENTS_FORM_INPUT)

# Calculate input dimensionality
N_FEATURES = np.shape(X)[1]  # data features/dimensionality
INPUT_DIM = N_FEATURES * N_SEGMENTS_FORM_INPUT if GROUP_SEGMENTS_FORM_INPUT == True else N_FEATURES

# if use of RNN and group segments to form timesteps inputs, reshape the X to [timesteps,features]. timesteps = N_SEGMENTS_FORM_INPUT
if(GROUP_SEGMENTS_FORM_INPUT == True and MODEL == "RNN"):
    FEATURES_ORIGINAL_SIZE = int(N_FEATURES / N_SEGMENTS_FORM_INPUT)
    X = X.reshape([-1, N_SEGMENTS_FORM_INPUT, FEATURES_ORIGINAL_SIZE])

print("X shape", np.shape(X))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# select neural network
# wrap the TF model with KerasClassifier()
print('Creating the model...')
if (MODEL == "FC"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build, input_dim=INPUT_DIM)
elif (MODEL == "TCN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=TCNNetwork.build, input_dim=INPUT_DIM)
elif (MODEL == "RNN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=RNNNetwork.build, timesteps=N_SEGMENTS_FORM_INPUT, input_dim=FEATURES_ORIGINAL_SIZE)
elif (MODEL == "CNN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=CNNNetwork.build, input_dim=INPUT_DIM)

# prepare the parameters' grid
learning_rate = [0.001]
init = ['GlorotUniform']
epochs = [10, 20]
batches = [8, 16]
dropout = [0.2, 0.5]
hidden_size = [8, 16, 32]
param_grid = dict(learning_rate=learning_rate, epochs=epochs, init=init, batch_size=batches, dropout=dropout, hidden_size=hidden_size)

# Because of imbalanced data, calculate class weights
class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
print("Class weights: ", class_weights)

# run the grid search and cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=CV)
grid_result = grid.fit(X, y, class_weight=class_weights)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# write the best model params to a JSON file
PARENT_DIR = "./CV_results/" + MODEL + '/' + 'chb{:02d}'.format(PATIENT)
os.makedirs(PARENT_DIR, exist_ok=True)  # create any parent directory that does not exist
best_model_file = open(PARENT_DIR + '/preictal_' + str(int(PREICTAL_DURATION / 60)) + "_best_params.json",
                       "w")
best_model_file.write(json.dumps(grid_result.best_params_))
best_model_file.close()
