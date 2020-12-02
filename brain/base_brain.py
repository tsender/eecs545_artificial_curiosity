import tensorflow as tf
import numpy as np
from PIL import Image
import os
import math
from typing import List, Callable

from memory import Memory
from experience import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BaseBrain:
    def __init__(self, nov_thresh: float, novelty_loss_type: str, max_train_epochs: int = 100):
        """Initializes the Brain by creating CNN and AE
        
        Params
        ------
        nov_thresh : float
            The novelty cutoff used in training
        novelty_function: Callable
            The callback that will be used to determine the novelty for any given feature-vector/reconstructed-vector pairs
        max_train_epochs: int
            Maximum number of training epochs (in case avg loss is still not at novelty thresh)
        """
        
        assert max_train_epochs > 0

        self._max_train_epochs = max_train_epochs
        self.memory = Memory() # The brain should handle the memory module
        self.nov_thresh = nov_thresh
        self._learning_session = 1
        self._loss_functions = { \
            "mae": tf.keras.losses.MeanAbsoluteError(), \
            "mse": tf.keras.losses.MeanSquaredError(), \
        }

        if novelty_loss_type.lower() not in self._loss_functions:
            print("Novelty loss type not recognized. Exiting.")
            exit(1)

        self.novelty_function = self._loss_functions[novelty_loss_type.lower()]

        # Parameters so we don't have any "magic numbers"
        self._batch_size = 4
        self._image_width = 224
        
        # Create models and optimizer
        self._init_CNN()
        self._init_AE()
        self._AE_opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

        print("Initialized Brain")

    def _init_CNN(self):
        """Initialize the Convolutional Neural Network"""

        # Create the base CNN model
        # TODO: Use different CNN base?
        self._CNN_Base = tf.keras.applications.VGG16(include_top=True)
        self._CNN_Base.trainable = False
        self._CNN = tf.keras.Model(self._CNN_Base.input, self._CNN_Base.layers[-1].input) # Use last FC layer as output

    def _init_AE(self):
        """Initialize the Auto Encoder"""

        # VGG FC layers use 4096 neurons
        input_vec = tf.keras.layers.Input(shape=(4096,))

        # Encoder
        # TODO: Maybe try LeakyRelu(alpha=0.2) for all activations
        e = tf.keras.layers.Dense(4096, 'relu')(input_vec)
        e = tf.keras.layers.Dense(1024, 'relu')(e)
        e = tf.keras.layers.Dense(256, 'relu')(e)
        e = tf.keras.layers.Dense(16, 'relu')(e)

        # Decoder
        d = tf.keras.layers.Dense(16, 'relu')(e)
        d = tf.keras.layers.Dense(256, 'relu')(d)
        d = tf.keras.layers.Dense(1024, 'relu')(d)
        output = tf.keras.layers.Dense(4096, 'relu')(d)

        self._AE = tf.keras.Model(input_vec, output)

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

        rgb_grain = Image.new("RGB", grain_in.size)
        rgb_grain.paste(rgb_grain)
        rgb_grain = tf.keras.preprocessing.image.img_to_array(rgb_grain)
        rgb_grain = tf.image.per_image_standardization(rgb_grain) # Transform images to zero mean and unit variance
        rgb_grain = tf.image.resize(rgb_grain, (self._image_width, self._image_width)) # Resize to CNN base input size
        return rgb_grain

    def add_grains(self, grains: List[Image.Image]):
        """Add new grains to memory

        Params:
            grains: List[Image.Image]
                List of new grains

        Returns:
            List of novelty for new grains
        """

        print("Adding new grains to memory...")
        assert len(grains) == 4 # Currently, we only allow 4 grains
        nov_list = []

        for g in grains:
            gtf = self._grain_to_tensor(g)
            gtf = tf.reshape(gtf, (1, gtf.shape[0], gtf.shape[1], gtf.shape[2])) # Reshape to (1,H,W,C)
            fvec = self._CNN(gtf)
            pred_fvec = self._AE(fvec)
            nov = self.novelty_function(fvec, pred_fvec).numpy()
            nov_list.append(nov)
            self.memory.push(Experience(nov, fvec.numpy().flatten(), g)) # Add to memory, fvec MUST be flattened
            
        return nov_list

    def evaluate_novelty(self, grains: List[Image.Image]):
        """Evaluate novelty of a list of grains

        Params:
            grains: List[Image.Image]
                List of new grains

        Returns:
            List of novelty for new grains
        """

        print("Evaluating grain novelty...")
        assert grains != [] and grains is not None
        nov_list = []

        for g in grains:
            gtf = self._grain_to_tensor(g)
            gtf = tf.reshape(gtf, (1, gtf.shape[0], gtf.shape[1], gtf.shape[2])) # Reshape to (1,H,W,C)
            fvec = self._CNN(gtf)
            pred_fvec = self._AE(fvec)
            nov = self.novelty_function(fvec, pred_fvec).numpy()
            nov_list.append(nov)
            
        return nov_list

    @tf.function
    def _train_step_ae(self, fvec: tf.Tensor):
        """Performs a single training step for the AE.

        Params:
            fvec: tf.Tensor
                A batch of feature vectors of size (batch, feature_dim) for trainng the AE
        
        Returns:
            The training loss for this step
        """

        with tf.GradientTape() as tape:
            predicted = self._AE(fvec, training=True)
            loss = self.novelty_function(fvec, predicted)
        
        gradients = tape.gradient(loss, self._AE.trainable_variables)
        self._AE_opt.apply_gradients(zip(gradients, self._AE.trainable_variables))
        return loss

    def learn_grains(self):
        """Train the AE to learn new features from memory"""

        print("Learning grains: Session %i" % self._learning_session)
        memList = self.memory.memList()
        fvecs = list(map(lambda e: e.featureVector, memList))
        dataset = tf.data.Dataset.from_tensor_slices(fvecs).shuffle(self._batch_size).repeat().batch(self._batch_size)
        dataset = iter(dataset)

        num_batches = math.ceil(len(memList) / self._batch_size)
        cur_avg_loss = float('inf')
        epoch = 0

        while cur_avg_loss >= self.nov_thresh:
            cur_avg_loss = 0
            for _ in range(num_batches):
                data = dataset.next()
                loss = self._train_step_ae(data).numpy()
                cur_avg_loss += (loss/num_batches)
            epoch += 1

            if epoch >= self._max_train_epochs:
                print(F"Breaking out of training loop. Current avg loss {cur_avg_loss:.4f} still greater than threshold of {self.nov_thresh:.4f}")
                break
        
        print("Learned in %i epochs" % epoch)
        self._learning_session += 1

if __name__ == "__main__":
    im = Image.open('data/x.jpg')
    brain = Brain(0.25, 'MSE') # 0.25 seems to be the smallest reasonable value for novelty thresh
    grain_nov = brain.add_grains([im, im, im, im])
    print("Grain novelty (before): ", grain_nov)
    brain.learn_grains()
    grain_nov = brain.evaluate_novelty([im, im, im, im])
    print("Grain novelty (after): ", grain_nov)
