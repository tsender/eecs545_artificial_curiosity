"""
The novelty module creates functions to evaluate the novelty (loss) which
are used in brain. The nevelty functions operate on the results of autoencoder
novelty func. 1: l1_norm loss
novelty func. 2: l2_norm loss
novelty func. 3: 
"""

import numpy as np
import tensorflow as tf

def l1_loss(feature_vector, reconstructed_vector):
    return tf.keras.losses.MAE(feature_vector, reconstructed_vector).numpy().item()

def l2_loss(feature_vector, reconstructed_vector):
    return tf.keras.losses.MSE(feature_vector, reconstructed_vector).numpy().item()