#!/usr/bin/env python3
""" Save the model """
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    save_best=False,
    filepath=None,
    verbose=True,
    shuffle=False,
):
    """ Saves the best iteration of the model """

    callbacks = []

    if early_stopping and validation_data:
        callbacks.append(
            K.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=patience)
        )

    if learning_rate_decay and validation_data:

        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                           verbose=1))

    if save_best and filepath:
        callbacks.append(
            K.callbacks.ModelCheckpoint(
                filepath, save_best_only=True, monitor="val_loss",
                mode="min"
            )
        )

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    return history
