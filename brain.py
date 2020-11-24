import tensorflow as tf
from memory import Memory
from experience import Experience
from map import Map
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from typing import List, Callable
from artificial_curiosity_types import ArtificialCuriosityTypes as act
from novelty import l1_loss, l2_loss
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Brain:
    def __init__(self, nov_thresh: float, novelty_function: Callable):
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

        self._AE_loss_object = keras.losses.MeanSquaredError()
        # ToDo: Ask ted about the optimization function
        self._AE_optimizer = tf.keras.optimizers.SGD(learning_rate=10)

    def _AE_loss(self, feature_vector: act.FeatureVector):
        reconstructed = self._AE(feature_vector)
        return self._AE_loss_object(y_true=feature_vector, y_pred=reconstructed)

    def _AE_grad(self, feature_vector: act.FeatureVector):
        with tf.GradientTape() as tape:
            loss_value = self._AE_loss(feature_vector)
            return loss_value, tape.gradient(loss_value, self._AE.trainable_variables)

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
        self._CNN = keras.Model(
            inputs=self._CNN_Base.layers[0].input, outputs=self._CNN_Base.layers[-2].input)

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

    def _prep_grains(self, grains: List[act.Grain], batch_size: int):
        """Prepares the grains used for finding directions. We have a specific method for this because the batch will always be zero
        and because I haven't written anything else yet

        Params
        ------

        grains : act.Grains
            The images that need to be processed
        batch_size : int
            The size of the batches that will be prepared

        Return
        ------

        A dataset split into a batch size of four

        """

        assert batch_size != None

        prep_grains = list(map(self._prep_single_grain, grains))
        return tf.data.Dataset.from_tensor_slices(prep_grains).batch(batch_size)

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
        vect_images = self._prep_grains(grains, 4)
        # Generate feature vectors
        feature_vects = self._CNN(list(iter(vect_images))[0])
        # Run the vectors through the auto-encoder to get reconstruction vectors
        reconstructed = self._AE(feature_vects)
        # Return the pairs of feature/reconstructed
        return list(zip(feature_vects, reconstructed))

    def eval_novelty(self, grains: List[act.Grain]):
        # This is because we haven't decided to allow different numbers of grains yet
        assert len(grains) == 4
        pairs = self._run_through_main_model(grains)
        loss = [Experience(l1_loss(*pairs[x]), pairs[x][0], grains[x])
                for x in range(len(grains))]
        return loss

    def learn_grains(self, exper: List[Experience], memory: Memory = None):

        # Const for batch size
        batch_size = 4

        # Create new list so we can reference old one later
        newList = exper
        if memory != None:
            newList += memory.memList()

        # Extract feature vectors and create batches
        feature_vects = list(map(lambda g: g.featureVector, newList))
        batches = tf.data.Dataset.from_tensor_slices(
            feature_vects).batch(batch_size)

        # create an object to hold the loss so we can check it as we go
        current_loss = 1000
        last_loss = 1000  # some large number
        diff = self.nov_thresh

        num_batches = math.ceil(len(feature_vects)/batch_size)

        # Here is where we do the actual training
        while diff >= self.nov_thresh:
            last_loss = current_loss
            current_loss = 0
            for batch in iter(batches):
                loss_value, grads = self._AE_grad(batch)
                self._AE_optimizer.apply_gradients(
                    zip(grads, self._AE.trainable_variables))
                current_loss += loss_value

            current_loss /= num_batches
            diff = last_loss - current_loss
            print(diff)

        # Put grains in long-term memory
        for e in exper:
            memory.push(e)


if __name__ == "__main__":
    im = Image.open('x.jpg')
    
    m = Memory()
    testing_brain = Brain(0.005, m)

    lst = testing_brain.eval_novelty([im, im, im, im])
    print(list(map(lambda x : x.novelty, lst)))

    for e in lst:
        m.push(e)

    testing_brain.learn_grains(lst, m)

    lst2 = testing_brain.eval_novelty([im, im, im, im])
    print(list(map(lambda x : x.novelty, lst2)))
