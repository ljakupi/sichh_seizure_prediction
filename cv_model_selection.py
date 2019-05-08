import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

import config
from data_processing.data_loader import load_data
from models.FC import FCNetwork

# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Init the config
cfg = config.Config(data_path='./data_features/CHB-MIT_features', NN='FC', patient=int(2))

DATA_DIRECTORY = './datasets/CHB-MIT/chb02/processed_data/preictal-' + str(int(cfg.preictal_duration / 60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']

print("Loading the data")
X, y = load_data(DATA_DIRECTORY, FILE_NAMES)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# select neural network
# wrap the TF model with KerasClassifier()
print('Creating the model...')
if (cfg.NN == "FC"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build)
elif (cfg.NN == "TCN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build)
elif (cfg.NN == "RNN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build)
elif (cfg.NN == "CNN"):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=FCNetwork.build)

# prepare the parameters' grid
learning_rate = [0.0001]
init = ['GlorotUniform']
epochs = [10]
batches = [8]
dropout = [0.2]
hidden_size = [16]
param_grid = dict(learning_rate=learning_rate, epochs=epochs, init=init, batch_size=batches, dropout=dropout,
                  hidden_size=hidden_size)

# run the grid search and cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# write the best model params to a JSON file
best_model_file = open("./CV-results/results.json", "w")  # write mode
best_model_file.write(grid_result.best_params_)
best_model_file.close()
