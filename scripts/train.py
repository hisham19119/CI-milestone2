import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from scripts.preprocess import normalize_image # type: ignore


def load_data(data_dir, target_size=(64, 64)):
    """Loads and preprocesses data from a directory in batches."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    data = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode=None,
        shuffle=True
    )

    normalized_data = []
    for batch in data:
        normalized_batch = [(img - img.mean()) / img.std() for img in batch]
        normalized_data.extend(normalized_batch)
        if len(normalized_data) >= data.samples:  # Stop once all samples are processed
            break

    return np.array(normalized_data)



def train_model(encoder, decoder, train_data, val_data, epochs=10):
    """Trains the encoder-decoder model."""
    autoencoder = tf.keras.Model(encoder.input, decoder(encoder.output))
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=epochs,
        batch_size=32
    )
    return history


def cross_validate(train_data, val_data, model_fn, configs):
    """Performs cross-validation for different model configurations."""
    best_config = None
    best_loss = float('inf')

    for config in configs:
        print(f"Training with config: {config}")
        encoder, decoder = model_fn(config)
        history = train_model(encoder, decoder, train_data, val_data, epochs=10)
        val_loss = min(history.history['val_loss'])

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config

    print(f"Best config: {best_config} with loss: {best_loss}")
    return best_config