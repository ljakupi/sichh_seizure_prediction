# This TCN implementation is heavily motivated from: https://github.com/philipperemy/keras-tcn/blob/master/tasks/imdb_tcn.py

import numpy as np
import tensorflow as tf


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.05):
    """Defines the residual block for the WaveNet TCN
    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = tf.keras.layers.Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        # x = BatchNormalization()(x)  # TODO should be WeightNorm here.
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = tf.keras.layers.Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = tf.keras.layers.add([prev_x, x])
    return res_x, x


class TCN:
    """Creates a TCN layer.
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2, # 3
                 nb_stacks=3, # 2
                 dilations=[1, 2, 4], # 1, 2, 4, 8, 16, 32, 64
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 name="TCN"):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        print('x', np.shape(x))
        # 1D FCN.
        x = tf.keras.layers.Convolution1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             dropout_rate=self.dropout_rate)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = tf.keras.layers.Lambda(lambda tt: tt[:, -1, :])(x)
        return x


class TCNNetwork():
    @staticmethod
    def build_model(
            input_dim=None,
            timesteps=None,
            learning_rate=0.0001,
            dropout=0.05,
    ):

        i = tf.keras.Input(shape=(timesteps, input_dim))
        x = TCN(nb_filters=128,
                kernel_size=2,
                nb_stacks=2,
                dilations=[1, 2, 4])(i) # 1, 2, 4, 8, 16, 32, 64
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[i], outputs=[x])

        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model
