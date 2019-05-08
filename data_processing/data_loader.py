import os
import config
import numpy as np
from itertools import zip_longest

# Init the config
cfg = config.Config(data_path='./data_features/CHB-MIT_features', NN='FC', patient=int(2))


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


def load_data(DATA_DIRECTORY, FILE_NAMES):
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
    print('All the data: ', np.shape(all_labeled_data))

    # Get the data and the labels
    X, y = all_labeled_data[:, :-1], all_labeled_data[:, -1]
    y = y.astype(int)  # cast the labels to int64

    return X, y
