import tensorflow as tf


class FCNetwork():
    @staticmethod
    def build(
            learning_rate=None,
            init=None,
            dropout=None,
            hidden_size=None,
            input_dim=None
    ):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size,
                                  input_dim=input_dim,
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
