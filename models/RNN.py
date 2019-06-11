import tensorflow as tf


class RNNNetwork():
    @staticmethod
    def build_model(
            input_dim=None,
            timesteps=None,
            units1=None,
            units2=None,
            units3=None,
            dropout1=None,
            dropout2=None,
            dropout3=None,
            learning_rate=None,
            multi_layer=False,
            sgd_opt=False,
            moment=False,
            l2_1=False,
            l2_2=False,
            l2_3=False,
            kernel_init=None
    ):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units1, return_sequences=True, input_shape=(timesteps, input_dim), kernel_regularizer=tf.keras.regularizers.l2(l2_1), kernel_initializer=kernel_init))
        model.add(tf.keras.layers.Dropout(dropout1))
        model.add(tf.keras.layers.LSTM(units2, return_sequences=True if multi_layer else False, kernel_regularizer=tf.keras.regularizers.l2(l2_2), kernel_initializer=kernel_init))
        model.add(tf.keras.layers.Dropout(dropout2))

        if multi_layer:
            model.add(tf.keras.layers.LSTM(units3, return_sequences=False,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_3),
                                           kernel_initializer=kernel_init))
            model.add(tf.keras.layers.Dropout(dropout3))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        if sgd_opt:
            # Define the optimizer and compile the model
            opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=moment)
        else:
            # Define the optimizer and compile the model
            opt = tf.keras.optimizers.Adam(lr=learning_rate)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

        return model