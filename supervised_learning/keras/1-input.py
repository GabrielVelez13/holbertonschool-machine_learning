#!/usr/bin/env python3
""" This module builds a neural network with Keras library """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with Keras library """

    assert len(layers) == len(activations)

    # Define the input layer
    inputs = K.Input(shape=(nx,))

    # Initialize L2 regularization. This is a form of weight decay that
    # encourages the model to have small weights, which helps prevent
    # overfitting.
    regularization = K.regularizers.l2(lambtha)

    # Define the first layer witch is connected to the input layer
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=regularization)(inputs)

    # Add the rest of the layers
    for i in range(1, len(layers)):
        # Apply dropout before adding the next layer. This randomly
        # sets a fraction '1 - keep_prob'
        # of the input units to 0 at each update during training time,
        # which helps prevent overfitting.
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=regularization)(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
