# import tensorflow as tf
# from tensorflow.keras import layers  # type: ignore

# def convolutional_autoencoder(input_shape):
#     """Defines a Convolutional Autoencoder."""
#     # Encoder
#     encoder_input = tf.keras.Input(shape=input_shape)
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#     # Decoder
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#     encoder = tf.keras.Model(encoder_input, encoded, name="encoder")
#     decoder = tf.keras.Model(encoded, decoded, name="decoder")
#     return encoder, decoder

# def vanilla_autoencoder(input_shape):
#     """Defines a Vanilla Autoencoder."""
#     # Encoder
#     encoder_input = tf.keras.Input(shape=input_shape)
#     x = layers.Flatten()(encoder_input)
#     x = layers.Dense(128, activation='relu')(x)
#     encoded = layers.Dense(64, activation='relu')(x)

#     # Decoder
#     decoder_input = tf.keras.Input(shape=(64,))
#     x = layers.Dense(128, activation='relu')(decoder_input)
#     x = layers.Dense(tf.math.reduce_prod(input_shape), activation='sigmoid')(x)
#     decoded = layers.Reshape(input_shape)(x)

#     encoder = tf.keras.Model(encoder_input, encoded, name="encoder")
#     decoder = tf.keras.Model(decoder_input, decoded, name="decoder")
#     return encoder, decoder

# def variational_autoencoder(input_shape):
#     """Defines a Variational Autoencoder."""
#     # Encoder
#     encoder_input = tf.keras.Input(shape=input_shape)
#     x = layers.Flatten()(encoder_input)
#     x = layers.Dense(128, activation='relu')(x)
#     z_mean = layers.Dense(64, name='z_mean')(x)
#     z_log_var = layers.Dense(64, name='z_log_var')(x)

#     def sampling(args):
#         z_mean, z_log_var = args
#         epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#     z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

#     # Decoder
#     decoder_input = tf.keras.Input(shape=(64,))
#     x = layers.Dense(128, activation='relu')(decoder_input)
#     x = layers.Dense(tf.math.reduce_prod(input_shape), activation='sigmoid')(x)
#     decoded = layers.Reshape(input_shape)(x)

#     encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
#     decoder = tf.keras.Model(decoder_input, decoded, name="decoder")
#     return encoder, decoder







import tensorflow as tf
from tensorflow.keras import layers  # type: ignore

def convolutional_autoencoder(input_shape):
    """Defines a Convolutional Autoencoder."""
    # Encoder
    encoder_input = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = tf.keras.Model(encoder_input, encoded, name="encoder")
    decoder = tf.keras.Model(encoded, decoded, name="decoder")
    return encoder, decoder

def vanilla_autoencoder(input_shape):
    """Defines a Vanilla Autoencoder."""
    # Encoder
    encoder_input = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(encoder_input)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(64, activation='relu')(x)

    # Decoder
    decoder_input = tf.keras.Input(shape=(64,))
    x = layers.Dense(128, activation='relu')(decoder_input)
    x = layers.Dense(tf.math.reduce_prod(input_shape), activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(x)

    encoder = tf.keras.Model(encoder_input, encoded, name="encoder")
    decoder = tf.keras.Model(decoder_input, decoded, name="decoder")
    return encoder, decoder

def variational_autoencoder(input_shape):
    """Defines a Variational Autoencoder."""
    # Encoder
    encoder_input = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(encoder_input)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(64, name='z_mean')(x)
    z_log_var = layers.Dense(64, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Decoder
    decoder_input = tf.keras.Input(shape=(64,))
    x = layers.Dense(128, activation='relu')(decoder_input)
    x = layers.Dense(tf.math.reduce_prod(input_shape), activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(x)

    encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    decoder = tf.keras.Model(decoder_input, decoded, name="decoder")
    return encoder, decoder
