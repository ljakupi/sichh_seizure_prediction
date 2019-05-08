import config
import tensorflow as tf


class TCNNetwork():
    @staticmethod
    def build(learning_rate=0.001, init='glorot_uniform', dropout=0.5, hidden_size=16):

        cfg = config.Config(data_path='./data_features/CHB-MIT_features', NN='FC', patient=int(2))

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size,
                                  input_dim=cfg.N_features * cfg.num_inputs if cfg.stack_segments_input == True else cfg.N_features,
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                  kernel_initializer=init),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        ])

        # Define the optimizer and compile the model
        opt_adam = tf.keras.optimizers.Adam(lr=learning_rate)

        model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])

        return model
