import tensorflow as tf
from Memory import Manory
from Experience import Experience
from map import Map
import novelty
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Brain():
    """
    The Brain class analysis novelty
    Attributes:
        - grains: a list of PIL images from map
        - novThresh: novelty threshold
        - novelty_function: loss function to evaluate novelty
    
    Methods:
        - CNN: will use a pre-trained CNN to perform the feature extraction (without training)
        - AE_arch: build the architechture of autoencoder
        - AE_compute: load trained autoencoder and compute reconstructed_vector
        - eval_novelty: evaluates the novelty (loss) based on the input/output of Autoencoder
        - learn_grains: performs the training of Autoencoder based on memory and new feature
        - train_AE: manually train the autoencoder
    """

    def __init__(self, novThresh: float, grains: list, novelty_function: novelty):
        """
        Parameters: 
            - novThresh: The cutoff for the change in novelty while training
            - novelty_function: Call a novelty funtion from the novelty module
            - grains: a list of PIL image from map module
        Returns: Initialization an object
        """
        self.novThresh = novThresh
        self.novelty_function = novelty_function
        self.grains = grains

    def CNN(self, grains):
        """
        ResNet50V2 (cut the output layer) is used to extract features
        feature dim --> 2048
        Parameters:
            - grains: (batch, height, weight) grey imgs from map
            reshape to (batch, 224, 224, 3) to match the ResNet50V2 model input
        Returns:
            - feature vector of each image (2048 dimensional) --> tensor 
        """
        # The following function automatically add the color channel
        # (heigh, width) --> (height, width, 3)
        grain_images = keras.preprocessing.image.img_to_array(grains)
        grain_images = tf.image.per_image_standardization(grain_images)
        # (batch, height, weight, 3) --> (batch, 224, 224, 3)
        grain_images = tf.image.resize_with_crop_or_pad(grain_images, 224, 224)

        base_model = keras.applications.ResNet50V2(
            include_top=True)  # load model
        base_model.trainable = False
        base_inputs = model.layers[0].input
        base_outputs = model.layers[-2].output

        transfer_CNN = keras.Model(
            inputs=base_inputs, outputs=base_outputs)  # new model
        feature_vector = transfer_CNN(grain_image)

        return feature_vector  # a tensor

    def AE_arch(self):
        """
        create the architecture of autoencoder
        """
        # encoder
        # ResNet50 output feature dim is 2048
        input_vec = layers.Input(shape=(2048,))
        encoder1 = layers.Dense(1024, 'sigmoid')(input_vec)
        encoder2 = layers.Dense(512, 'sigmoid')(encoder1)
        # decoder
        decoder1 = layers.Dense(1024, 'sigmoid')(encoder2)
        decoder2 = layers.Dense(2048, 'sigmoid')(decoder1)

        # build model
        autoencoder = Model(inputs=input_vec, outputs=decoder2)

        return autoencoder

    def AE_compute(self, feature_vector, trained_model=True):
        """
        Parameters: 
            - feature_vector: CNN output (e.g., 2048 dim tensor)
            - trained_model: train model/load saved model before compute reconstructed_vector
        Returns:
            - reconstructed_vector 
        """
        # create checkpoint so that autoencoder can be saved and loaded
        checkpoint_path = "Autoencoder/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if saved_model:
            autoencoder = self.AE_arch()
            autoencoder.load_weights(checkpoint_path)
            reconstructed_vector = autoencoder(feature_vector)

        else:
            print("No trained autoencoder found")

        return reconstructed_vector

    def learn_grains(self, feature_vector, memory=None):
        """
        learn_grains trains the autoencoder to memorize the feature vector of new grains
        Parameters:
            - feature_vector: CNN output (e.g., 2048 dim tensor)
            - memory: 
        Returns:
            - void --> updating autoencoder
        """
        # # save the model to check
        # checkpoint_path = "Autoencoder/cp.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        # cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                               save_weights_only=True,
        #                                               verbose=0)

        # # prepare data for training: both new feature_vectors and from memory

        # # Call train_AE function to train autoencoder
        # autoencoder = self.AE_arch()
        # self.train_AE(autoencoder, train_dataset, callbacks=[cp_callback],
        #               epochs=10, learning_rate=0.01,)

        pass

    def eval_novelty(self, grains: list):
        """
        Parameters:
            - grains: a list of PIL images
        Returns:
            - novelty(loss) -- list float
        """
        loss_list = []
        for grain in self.grains:
            feature_vector = self.CNN(grain)
            reconstructed_vector = self.AE_compute(
                feature_vector, trained_model=True)
            loss = self.novelty_function(feature_vector, reconstructed_vector)
            loss_list.append(loss)

        return loss_list

    def train_AE(self, model, train_dataset, callbacks, epochs=10, learning_rate=0.01):
        """
        Manually train autoencoder
        Parameters:
            - model: model to train. In this case, it is autoencoder
            - train_dataset: new feature_vector and memory
            - epochs, learning_rate
            - callbacks: save the model to check
        Return:
            - void: train the autoencoder
        """
        # optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # loss_fn = keras.losses.MSE(vec_true, vec_pred) # could be changed

        pass


# if __name__ = "__main__":
#     # init brain class
#     novThresh = 0.1
#     grains = Map().get_fov((0,0)) # how to load grains properly???
#     novelty_function = novelty.l1_loss

#     brain = Brain(novThresh, grains, novelty_function)
