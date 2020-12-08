import tensorflow as tf
import numpy as np
from PIL import Image
import os
import math
from typing import List, Tuple

from memory import BaseMemory
from experience import Experience
import networks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Brain:
    def __init__(self, memory: BaseMemory, img_size: Tuple, nov_thresh: float = 0.25, 
                novelty_loss_type: str = 'MSE', train_epochs_per_iter: int = 1, learning_rate: float = 0.001):
        """Initializes the Brain by creating CNN and AE
        
        Args:
            memory: BaseMemory
                A memory object that implements BaseMemory  (such as PriorityBasedMemory)
            img_size: Tuple
                The image size of each grain from the agent's field of view
            nov_thresh : float
                (Currently deprecated). The novelty cutoff used in training
            novelty_loss_type: str
                A string indicating which novelty function to use (MSE or MAE)
            train_epochs_per_iter: int
                Number of epochs to train for in a single training session
            learning_rate: float
                Learning rate for neural network optimizer
        """
        
        assert train_epochs_per_iter > 0

        self._memory = memory
        self._img_size = img_size
        self._train_epochs_per_iter = train_epochs_per_iter
        self._nov_thresh = nov_thresh
        self._batch_size = 4
        self._novelty_loss_type = novelty_loss_type
        self._learning_rate = learning_rate

        self._loss_functions = { \
            "mae": tf.keras.losses.MeanAbsoluteError(), \
            "mse": tf.keras.losses.MeanSquaredError(), \
        }

        if novelty_loss_type.lower() not in self._loss_functions:
            print("Novelty loss type not recognized. Exiting.")
            exit(1)

        self.novelty_function = self._loss_functions[novelty_loss_type.lower()]

        # Create network and optimizer
        self._network = networks.create_network(img_size)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # print("Initialized Brain")

    def get_name(self):
        """Returns the full descriptive name of the brain object.
        
        Returns
            The name of the brain object as a string
        """
        name_str = "Brain"
        name_str += "_" + self._memory.get_name() 
        name_str += "_ImgSize" + str(self._img_size[0])
        name_str += "_Nov" + self._novelty_loss_type.upper()
        name_str += "_Train" + str(self._train_epochs_per_iter)
        name_str += "_Lrate" + str(self._learning_rate)
        return name_str

    # def _init_CNN(self):
    #     """Initialize the Convolutional Neural Network"""

    #     # Create the base CNN model
    #     # TODO: Use different CNN base?
    #     self._CNN_Base = tf.keras.applications.VGG16(include_top=True)
    #     self._CNN_Base.trainable = False
    #     self._CNN = tf.keras.Model(self._CNN_Base.input, self._CNN_Base.layers[-1].input) # Use last FC layer as output

    # def _init_AE(self):
    #     """Initialize the Auto Encoder"""

    #     # VGG FC layers use 4096 neurons
    #     input_vec = tf.keras.layers.Input(shape=(4096,))

    #     # Encoder
    #     # TODO: Maybe try LeakyRelu(alpha=0.2) for all activations
    #     e = tf.keras.layers.Dense(4096, 'relu')(input_vec)
    #     e = tf.keras.layers.Dense(1024, 'relu')(e)
    #     e = tf.keras.layers.Dense(256, 'relu')(e)
    #     e = tf.keras.layers.Dense(16, 'relu')(e)

    #     # Decoder
    #     d = tf.keras.layers.Dense(16, 'relu')(e)
    #     d = tf.keras.layers.Dense(256, 'relu')(d)
    #     d = tf.keras.layers.Dense(1024, 'relu')(d)
    #     output = tf.keras.layers.Dense(4096, 'relu')(d)

    #     self._AE = tf.keras.Model(input_vec, output)

    def _grain_to_tensor(self, grain_in: Image.Image):
        """Convert a single grain to a tf.Tensor

        Params
        ------
        grain_in : Image.Image
            An image from the rover's "camera" that needs to be preprocessed

        Return
        ------
            The grain as a tf.Tensor
        """

        # rgb_grain = Image.new("RGB", grain_in.size)
        # rgb_grain.paste(rgb_grain)
        # rgb_grain = tf.keras.preprocessing.image.img_to_array(rgb_grain)
        # rgb_grain = tf.image.per_image_standardization(rgb_grain) # Transform images to zero mean and unit variance
        # rgb_grain = tf.image.resize(rgb_grain, (self._image_width, self._image_width)) # Resize to CNN base input size
        tf_img = tf.keras.preprocessing.image.img_to_array(grain_in)
        tf_img = (tf_img - 127.5) / 127.5 # Normalize to [-1,1]
        tf_img = tf.reshape(tf_img, self._img_size)
        return tf_img

    def add_grains(self, grains: List[List[Image.Image]]):
        """Add new grains to memory

        Params:
            grains: List[List[Image.Image]]
                2D List of new grains

        Returns:
            2D List of novelty for new grains
        """

        # print("Adding new grains to memory...")
        assert len(grains) == 2 # Currently, we only allow 4 grains
        assert len(grains[0]) == 2 # Currently, we only allow 4 grains
        nov_list = []

        for row in grains:
            temp_nov = []
            for g in row:
                grain_tf = self._grain_to_tensor(g)
                grain_tf = tf.reshape(grain_tf, (1, grain_tf.shape[0], grain_tf.shape[1], grain_tf.shape[2])) # Reshape to (1,H,W,C)
                predicted_grain = self._network(grain_tf)
                nov = self.novelty_function(grain_tf, predicted_grain).numpy()
                temp_nov.append(nov)
                self._memory.push(Experience(nov, g))
            nov_list.append(temp_nov)
            
        return nov_list

    def evaluate_grains(self, grains: List[List[Image.Image]]):
        """Evaluate a list of grains

        Params:
            grains: List[List[Image.Image]]
                2D List of new grains

        Returns:
            2D List of novelty for new grains, and 2D list for reconstructed grains
        """

        # print("Evaluating grain novelty...")
        assert grains != [] and grains is not None
        nov_list = []
        pred_grains_list = []

        for row in grains:
            temp_nov = []
            temp_grains = []
            for g in row:
                grain_tf = self._grain_to_tensor(g)
                grain_tf = tf.reshape(grain_tf, (1, grain_tf.shape[0], grain_tf.shape[1], grain_tf.shape[2])) # Reshape to (1,H,W,C)
                predicted_grain = self._network(grain_tf)
                nov = self.novelty_function(grain_tf, predicted_grain).numpy()

                temp_nov.append(nov)
                pred_grain = tf.reshape(predicted_grain, (grain_tf.shape[1], grain_tf.shape[2], grain_tf.shape[3]))
                pred_grain = tf.keras.preprocessing.image.array_to_img((pred_grain * 127.5) + 127.5) # Convert back to [0,255]
                temp_grains.append(pred_grain)
            nov_list.append(temp_nov)
            pred_grains_list.append(temp_grains)
            
        return nov_list, pred_grains_list

    @tf.function
    def _train_step(self, images: tf.Tensor):
        """Performs a single training step for the network.

        Params:
            images: tf.Tensor
                A batch of images of size (batch, height, width, channel) for trainng the network
        
        Returns:
            The training loss for this step
        """

        with tf.GradientTape() as tape:
            predicted = self._network(images, training=True)
            loss = self.novelty_function(images, predicted)
        
        gradients = tape.gradient(loss, self._network.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))
        return loss

    def learn_grains(self):
        """Train the network to learn new features from memory
        
        Returns:
            The current average loss from the last training epoch
        """

        memory_list = self._memory.as_list()
        grains = list(map(lambda e: self._grain_to_tensor(e.grain), memory_list))
        dataset = tf.data.Dataset.from_tensor_slices(grains).shuffle(self._batch_size).batch(self._batch_size).repeat()
        dataset = iter(dataset)

        num_batches = math.ceil(len(memory_list) / self._batch_size)
        cur_avg_loss = 0

        for i in range(self._train_epochs_per_iter):
            cur_avg_loss = 0
            
            for j in range(num_batches):
                data = dataset.next()
                loss = self._train_step(data).numpy()
                cur_avg_loss += (loss/num_batches)
        
        return cur_avg_loss

if __name__ == "__main__":
    from memory import PriorityBasedMemory, ListBasedMemory

    img = Image.open('data/x.jpg').convert('L').resize((64,64))

    # NOTE
    # 0.25 seems to be the smallest value that the novelty loss will go.
    # If we use nov_thresh for training, do not set below 0.25
    brain1 = Brain(ListBasedMemory(64), (64,64,1), 0.25, 'MSE', 1)
    brain2 = Brain(PriorityBasedMemory(64), (64,64,1), 0.25, 'MSE', 10, 0.001)

    print(brain2.get_name())
    grain_nov = brain2.add_grains([
        [img, img],
        [img, img]
    ])
    print("Grain novelty (before): ", grain_nov)
    loss = brain2.learn_grains()
    print(F"Loss: {loss}")
    grain_nov, _ = brain2.evaluate_grains([
        [img, img],
        [img, img]
    ])
    print("Grain novelty (after): ", grain_nov)
