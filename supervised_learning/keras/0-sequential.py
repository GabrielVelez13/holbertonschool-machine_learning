#!/usr/bin/env python3
""" starting out with Keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ A neural network using Keras """
    assert len(layers) == len(activations)

    # Initialize a sequential model
    model = K.models.Sequential()

    # Add each layer
    for i in range(len(layers)):
        # If it's the first layer, we need to specify the input_dim
        if i == 0:
            model.add(
                K.layers.Dense(
                    layers[i],
                    input_dim=nx,
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                )
            )

        # If it's not the last layer, add dropout
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
