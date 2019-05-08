import os
import config
import numpy as np
import tensorflow as tf
from itertools import zip_longest

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split


# Debugger to check whic device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Init the config
cfg = config.Config(data_path='./data_features/CHB-MIT_features', NN='FC', patient=int(2))

DATA_DIRECTORY = './datasets/CHB-MIT/chb02/processed_data/preictal-' + str(int(cfg.preictal_duration/60))
FILE_NAMES = ['interictal_segments.npy', 'preictal_segments.npy']



# Helper method to group several segments into one which represents one input (with class 0 or 1)
def segments_grouper(n_inputs, iterable, fillvalue=np.float64(0)):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n_inputs
    fv = [fillvalue] * len(iterable[0])
    return zip_longest(fillvalue=list(fv), *args)


# Helper method to label the segments interictal = 0, preictal = 1
def labeler(example, index):
    # return example, tf.cast(index, tf.int64)
    return np.append(example, np.int64(index))


# Load interictal and preictal segments from npy files using tf.data.Dataset loader
labeled_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
    loaded_segments = np.load(os.path.join(DATA_DIRECTORY, file_name))

    # If cfg.stack_segments_input, group several segments into one, to form one input
    if (cfg.stack_segments_input):
        loaded_segments = list(segments_grouper(cfg.num_inputs, loaded_segments))  # grouped segments

    # Annotate the segments
    labeled_dataset = list(map(lambda ex: labeler(ex, i), loaded_segments))
    labeled_data_sets.append(labeled_dataset)

print('Interictal segments: ', np.shape(labeled_data_sets[0]))
print('Preictal segments: ', np.shape(labeled_data_sets[1]))

# Merge the two segments' lists (interictal and preictal) into one
all_labeled_data = np.concatenate(labeled_data_sets)
print('Total data: ', np.shape(all_labeled_data))

# Get the data and the labels
X, y = all_labeled_data[:, :-1], all_labeled_data[:, -1]
y = y.astype(int)  # cast the labels to int64



# Split training and test data so we can use the test set at the very end to test the best estimator that GridSearchCV gives
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=y)



# Check how many data-points are annotated as 1 (preictal) or 2 (ictal) and 0 (interictal) respectively
unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))




# Define 10-fold cross validation
# skf = StratifiedKFold(n_splits=5, shuffle=True)
#
# total_cv_accuracy = []
# total_cv_loss = []
#
# # Function to create model, required for KerasClassifier
# def create_model(learning_rate = 0.001, init='glorot_uniform', dropout=0.5, hidden_size=cfg.num_hidden):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(hidden_size,
#                               input_dim=cfg.N_features * cfg.num_inputs if cfg.stack_segments_input == True else cfg.N_features,
#                               activation='relu',
#                               kernel_regularizer=tf.keras.regularizers.l2(0.1),
#                               kernel_initializer = init),
#         tf.keras.layers.Dropout(dropout),
#         tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.1))
#     ])
#
#     # Define the optimizer and compile the model
#     opt_adam = tf.keras.optimizers.Adam(lr=learning_rate)
#
#     model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])
#
#     return model

#
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
#
# print('Creating the model...')
# # create model
# model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)
#
# # grid search epochs, batch size and optimizer
# learning_rate = [0.0001]
# init = ['GlorotUniform']
# epochs = [10]
# batches = [8, 16]
# dropout = [0.2, 0.5, 0.8]
# hidden_size = [16, 32, 64]
# param_grid = dict(learning_rate=learning_rate, epochs=epochs, init=init, batch_size=batches, dropout=dropout, hidden_size=hidden_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(X_train, y_train)
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
#
# # Write-Overwrites
# best_model_file = open("./CV-results/results.txt","w")#write mode
# best_model_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# best_model_file.close()







# print('Running the cross-validation...')
# # evaluate using 10-fold cross validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# results = cross_val_score(model, X, y, cv=kfold)
#
# print('Cross-validation score result: ', results.mean())


#
# for index, (train_indices, test_indices) in enumerate(skf.split(X, y)):
#     # Get train and test data
#     x_train, x_test = X[train_indices], X[test_indices]
#     y_train, y_test = y[train_indices], y[test_indices]

# Define some hyper-parameters
BUFFER_SIZE = 500000
BATCH_SIZE = 8
TAKE_SIZE = 5000  # count to indicate how many samples to use for testing


# Load data to tf.data.Dataset, shuffle and create batches
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE, drop_remainder=False)

# Create the network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16,
                          input_dim=cfg.N_features * cfg.num_inputs if cfg.stack_segments_input == True else cfg.N_features,
                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), kernel_initializer = 'GlorotUniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.1))
])

# Define the optimizer and compile the model
opt_adam = tf.keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])

# Train the network
# Use validation_split so the model automatically splits a validation set from training set.
# Model is evaluated at the end of each training epoch.
model.fit(train_set, epochs=10) # validation_data=test_set # validation_split=0.3

print("Testing...")
# Evaluate the network
model.evaluate(test_set)

# print("Trained on fold " + str(index + 1) + "/10 | ", "%s: %.2f%%" % (model.metrics_names[1], eval_acc))
# total_cv_accuracy.append(eval_acc)
# total_cv_loss.append(eval_loss)
#
# print('\nEval loss: {:.2f}, Eval accuracy: {:.2f}'.format(np.mean(total_cv_loss), np.mean(total_cv_accuracy)))
