import tensorflow as tf
from Memory import Memory
from Experience import Experience
from Map import Map
import novelty
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from typing import List
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as act
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class New_Brain:
    def __init__(self, nov_thresh: float, novelty_function: novelty):
        """Initializes the Brain by creating CNN and AE
        
        Params
        ------
        nov_thresh : float
            The novelty cutoff used in training
        novelt_function
            The callback that will be used to determine the novelty for any given feature-vector/reconstructed-vector pairs
        
        """

        self.nov_thresh = nov_thresh
        self.novelty_function = novelty_function

        # Parameters so we don't have any "magic numbers"
        self._image_width = 224
        self._model_input = 4096
        self._ae_l1 = self._model_input ** (1/2)
        self._ae_l2 = self._model_input ** (1/2)

        # Create models
        self._init_CNN()
        self._init_AE()

    def _init_CNN(self):
        """Initialize the Convolutional Neural Network

        """

        # Create the base CNN model
        # ToDo: Might want to change out the model because this one has a lot of parameters
        self._CNN_Base = keras.applications.VGG16(include_top=True)
        # Freeze it
        self._CNN_Base.trainable = False
        # Combine the first layer and the second-to-last layer into a mode
        # We don't want the last layer because we don't want classification, we want a feature vector
        self._CNN = keras.Model(inputs=self._CNN_Base.layers[0].input, outputs=self._CNN_Base.layers[-2].input)

    def _init_AE(self):
        """Initialize the Auto Encoder

        """

        # vgg output is 4096
        input_vec = layers.Input(shape=(4096,))

        # encoder
        encoder1 = layers.Dense(self._model_input, 'sigmoid')(input_vec)
        encoder2 = layers.Dense(self._ae_l1, 'sigmoid')(encoder1)
        encoder3 = layers.Dense(self._ae_l2, 'sigmoid')(encoder2)

        # decoder
        decoder1 = layers.Dense(self._ae_l2, 'sigmoid')(encoder3)
        decoder2 = layers.Dense(self._ae_l1, 'sigmoid')(decoder1)
        decoder3 = layers.Dense(self._model_input, 'sigmoid')(decoder2)

        self._AE = Model(inputs=input_vec, outputs=decoder3)

    def _prep_single_grain(self, grain: act.Grain):
        """Processes a single grain for use in our model

        Params
        ------

        grain : act.Grain
            An image from the rover's "camera" that needs to be preprocessed


        Return
        ------

        A grain that has been reprocessed and is now a tensor

        """

        grain = keras.preprocessing.image.img_to_array(grain)
        grain = tf.image.per_image_standardization(grain)
        # Resize so that it fits instead of crops
        grain = tf.image.resize(grain, (self._image_width, self._image_width))
        # grain = tf.reshape(grain, (self._image_width, self._image_width, 3))

        return grain

    def _prep_directional_grains(self, grains: List[act.Grain]):
        """Prepares the grains used for finding directions. We have a specific method for this because the batch will always be zero
        and because I haven't written anything else yet

        Params
        ------

        grains : act.Grains
            The images that need to be processed

        Return
        ------

        A dataset split into a batch size of four

        """
        prep_grains = list(map(self._prep_single_grain, grains))
        return tf.data.Dataset.from_tensor_slices(prep_grains).batch(4)

    def _run_through_main_model(self, grains: List[act.Grain]):
        """This function is just an example of how to send grains through the entiere main model and get out
        pairs of feature/reconstruction vectors

        Params
        ------

        grains : act.Grains
            The images that will be sent through the model

        Return
        ------

        A list of the feature vector/reconstruced vector pairs from the grains. They are returned in the same ordeer that they were
        passed in

        """

        # Prepare the images for the CNN
        vect_images = self._prep_directional_grains(grains)
        # Generate feature vectors
        feature_vects = self._CNN(list(iter(vect_images))[0])
        # Run the vectors through the auto-encoder to get reconstruction vectors
        reconstructed = self._AE(feature_vects)
        # Return the pairs of feature/reconstructed
        return list(zip(feature_vects, reconstructed))

    def eval_novelty(self, grains: List[act.Grain]):
        pass

    def learn_grains(self, grains: List[act.Grain], memory: Memory = None):
        pass


if __name__ == "__main__":
    nb = New_Brain(0.5, None)
    im = Image.open("x.jpg")
    nb._run_through_main_model([im, im, im, im])
