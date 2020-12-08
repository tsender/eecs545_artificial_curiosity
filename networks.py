from typing import Tuple
import tensorflow as tf
import numpy as np

def create_network(image_size: Tuple):
    """Create the CNN-AE based on the input image size. Only square grey scale images allowed.
    The input and output sizes for the network are the same.

    Args:
        image_size: Tuple
            Image size as Tuple of (H,W,C)
    
    Returns:
        A tensorflow model for the network.
    """

    tf.random.set_seed(1234)
    allowed_sizes = [32, 64, 128]
    models = [create_network32, create_network64, create_network128]

    # Check image size is allowed
    assert image_size is not None and len(image_size) == 3
    assert image_size[0] == image_size[1]
    assert image_size[2] == 1
    assert image_size[0] in allowed_sizes

    # Get index and call function to create network
    idx = allowed_sizes.index(image_size[0])
    return models[idx]()

def create_network32():
    """Create the network for input size of (32, 32, 1)"""

    in_image = tf.keras.Input(shape=(32, 32, 1))
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Encoder
    e = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(in_image)
    e = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 16, 16, 64)
    e = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 8, 8, 64)
    e = tf.keras.layers.Flatten()(e) # Shape (1, 4096)
    e = tf.keras.layers.Dense(512, activation=leaky_relu)(e)
    e = tf.keras.layers.Dense(128, activation=leaky_relu)(e)
    e = tf.keras.layers.Dense(16, activation=leaky_relu)(e)

    # Decoder
    d = tf.keras.layers.Dense(16, activation=leaky_relu)(e)
    d = tf.keras.layers.Dense(128, activation=leaky_relu)(d)
    d = tf.keras.layers.Dense(512, activation=leaky_relu)(d)
    d = tf.keras.layers.Dense(8*8*64, activation=leaky_relu)(d)
    d = tf.keras.layers.Reshape((8,8,64))(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 16, 16, 64)
    d = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 32, 32, 64)
    d = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    d = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    out_image = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='tanh', kernel_initializer=tf.initializers.he_normal())(d)

    model = tf.keras.Model(in_image, out_image)
    return model

def create_network64():
    """Create the network for input size of (64, 64, 1)"""

    in_image = tf.keras.Input(shape=(64, 64, 1))
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Encoder
    e = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(in_image)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 32, 32, 32)
    e = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 16, 16, 64)
    e = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 8, 8, 128)
    e = tf.keras.layers.Flatten()(e) # Shape (1, 8192)
    e = tf.keras.layers.Dense(1024, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(e)
    e = tf.keras.layers.Dense(512, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(e)
    e = tf.keras.layers.Dense(128, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(e)

    # Decoder
    d = tf.keras.layers.Dense(128, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(e)
    d = tf.keras.layers.Dense(512, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(d)
    d = tf.keras.layers.Dense(1024, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(d)
    d = tf.keras.layers.Dense(8*8*128, activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.L2)(d)
    d = tf.keras.layers.Reshape((8,8,128))(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 16, 16, 128)
    d = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 32, 32, 128)
    d = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 64, 64, 64)
    d = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.L2)(d)
    out_image = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='tanh', kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.L2)(d)

    model = tf.keras.Model(in_image, out_image)
    return model

def create_network128():
    """Create the network for input size of (128, 128, 1)"""

    in_image = tf.keras.Input(shape=(128, 128, 1))
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Encoder
    e = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(in_image)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 64, 64, 32)
    e = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 32, 32, 32)
    e = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 16, 16, 64)
    e = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(e)
    e = tf.keras.layers.AveragePooling2D()(e) # Shape (B, 8, 8, 128)
    e = tf.keras.layers.Flatten()(e)
    e = tf.keras.layers.Dense(512, activation=leaky_relu)(e)
    e = tf.keras.layers.Dense(128, activation=leaky_relu)(e)
    e = tf.keras.layers.Dense(32, activation=leaky_relu)(e)

    # Decoder
    d = tf.keras.layers.Dense(32, activation=leaky_relu)(e)
    d = tf.keras.layers.Dense(128, activation=leaky_relu)(d)
    d = tf.keras.layers.Dense(512, activation=leaky_relu)(d)
    d = tf.keras.layers.Dense(8*8*128, activation=leaky_relu)(d)
    d = tf.keras.layers.Reshape((8,8,128))(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 16, 16, 128)
    d = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 32, 32, 128)
    d = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 64, 64, 64)
    d = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    d = tf.keras.layers.UpSampling2D()(d) # Shape (B, 128, 128, 32)
    d = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=leaky_relu, kernel_initializer=tf.initializers.he_normal())(d)
    out_image = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='tanh', kernel_initializer=tf.initializers.he_normal())(d)

    model = tf.keras.Model(in_image, out_image)
    return model

if __name__ == "__main__":
    model = create_network((64,64,1))
    model.summary()