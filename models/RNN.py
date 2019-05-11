import tensorflow as tf


class RNNNetwork():
    @staticmethod
    def build(
            learning_rate=None,
            init=None,
            dropout=None,
            hidden_size=None,
            input_dim=None,
            timesteps=None
    ):

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim), activation='relu', kernel_initializer=init),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(hidden_size, return_sequences=False, activation='relu', kernel_initializer=init),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Define the optimizer and compile the model
        opt_adam = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])

        return model
