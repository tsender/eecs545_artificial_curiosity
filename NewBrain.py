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
        self.nov_thresh = nov_thresh
        self.novelty_function = novelty_function
        self._init_CNN()

    def _init_CNN(self):
        self._CNN_Base = keras.applications.VGG16(include_top=True)
        self._CNN_Base.trainable = False
        self._CNN = keras.Model(
            inputs=self._CNN_Base.layers[0].input, outputs=self._CNN_Base.layers[-2].input)

    def _prep_single_image(self, grain):
        grain = keras.preprocessing.image.img_to_array(grain)
        grain = tf.image.per_image_standardization(grain)
        grain = tf.image.resize(grain, (224, 224))
        grain = tf.reshape(grain, (224, 224, 3))

        return grain

    # Go back and shuffle eventually
    def _prep_multi_image(self, grains):
        # right now set to 4, but will probably want more in the future
        return tf.data.Dataset.from_tensor_slices(list(map(self._prep_single_image, grains))).batch(4)

    def _CNN_process_grains(self, vect_images):
        return self._CNN(list(iter(vect_images))[0])

    def eval_novelty(self, grains: List[act.Grain]):
        pass

    def learn_grains(self, grains: List[act.Grain], memory: Memory = None):
        pass


if __name__ == "__main__":
    nb = New_Brain(0.5, None)
    im = Image.open("x.jpg")
    k = nb._prep_multi_image([im, im, im, im, im])
    nb._CNN_process_multi(nb._prep_multi_image([im, im, im, im]))
    list(iter(k))[0]
