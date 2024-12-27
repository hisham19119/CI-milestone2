# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore


# def load_data(data_dir, target_size=(64, 64), batch_size=32):
#     """Loads and preprocesses data from a directory in batches."""
#     datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#     data_generator = datagen.flow_from_directory(
#         data_dir,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode="input",  # Key update: "input" ensures x == y
#         shuffle=True
#     )
#     return data_generator


# def train_model(encoder, decoder, train_data, val_data, epochs=10):
#     """Trains the encoder-decoder model."""
#     # If encoder output is a tuple (e.g., in VAE), extract z (latent space)
#     if isinstance(encoder.output, (tuple, list)):
#         latent_space = encoder.output[-1]  # Use the z latent vector
#     else:
#         latent_space = encoder.output  # For vanilla/convolutional autoencoders

#     autoencoder = tf.keras.Model(encoder.input, decoder(latent_space))
#     autoencoder.compile(optimizer='adam', loss='mse')

#     # Autoencoder: Input equals target
#     history = autoencoder.fit(
#         train_data,
#         validation_data=val_data,
#         epochs=epochs
#     )
#     return history


# def cross_validate(train_data, val_data, model_fn, configs):
#     """Performs cross-validation for different model configurations."""
#     best_config = None
#     best_loss = float('inf')

#     for config in configs:
#         print(f"Training with config: {config}")
#         encoder, decoder = model_fn(config)
#         history = train_model(encoder, decoder, train_data, val_data, epochs=10)
#         val_loss = min(history.history['val_loss'])

#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_config = config

#     print(f"Best config: {best_config} with loss: {best_loss}")
#     return best_config










import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

def load_data(data_dir, target_size=(64, 64), batch_size=32):
    """Loads and preprocesses data from a directory in batches."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="input",  # Key update: "input" ensures x == y
        shuffle=True
    )
    return data_generator

def train_model(encoder, decoder, train_data, val_data, epochs=10):
    """Trains the encoder-decoder model."""
    # If encoder output is a tuple (e.g., in VAE), extract z (latent space)
    if isinstance(encoder.output, (tuple, list)):
        latent_space = encoder.output[-1]  # Use the z latent vector
    else:
        latent_space = encoder.output  # For vanilla/convolutional autoencoders

    autoencoder = tf.keras.Model(encoder.input, decoder(latent_space))
    autoencoder.compile(optimizer='adam', loss='mse')

    # Autoencoder: Input equals target
    history = autoencoder.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
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
