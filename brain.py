import tensorflow as tf
import numpy as np
import PIL
import os
import math
from typing import List, Callable

from memory import Memory
from experience import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Brain:
    def __init__(self, nov_thresh: float, novelty_loss_type: str):
        """Initializes the Brain by creating CNN and AE
        
        Params
        ------
        nov_thresh : float
            The novelty cutoff used in training
        novelty_function: Callable
            The callback that will be used to determine the novelty for any given feature-vector/reconstructed-vector pairs
        """

        self.memory = Memory()
        self.nov_thresh = nov_thresh

        if novelty_loss_type == 'MAE':
            self.novelty_function = tf.keras.losses.MeanAbsoluteError()
        elif novelty_loss_type == 'MSE':
            self.novelty_function = tf.keras.losses.MeanSquaredError()

        # Parameters so we don't have any "magic numbers"
        self._batch_size = 4
        self._image_width = 224
        self._fvec_size = 4096
        self._ae_l1_fmaps = self._fvec_size ** (1/2)
        self._ae_l2_fmaps = self._fvec_size ** (1/2)
        
        # Create models and optimizer
        self._init_CNN()
        self._init_AE()
        self._AE_opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

        print("Initialized Brain")

    def _init_CNN(self):
        """Initialize the Convolutional Neural Network"""

        # Create the base CNN model
        # TODO: Might want to change out the model because this one has a lot of parameters
        self._CNN_Base = tf.keras.applications.VGG16(include_top=True)
        self._CNN_Base.trainable = False
        self._CNN = tf.keras.Model(self._CNN_Base.input, self._CNN_Base.layers[-1].input) # Use last FC layer as output

    def _init_AE(self):
        """Initialize the Auto Encoder"""

        # VGG FC layers use 4096 neurons
        input_vec = tf.keras.layers.Input(shape=(4096,))

        # Encoder
        e = tf.keras.layers.Dense(self._fvec_size, 'relu')(input_vec)
        e = tf.keras.layers.Dense(self._ae_l1_fmaps, 'relu')(e)
        e = tf.keras.layers.Dense(self._ae_l2_fmaps, 'relu')(e)

        # Decoder
        d = tf.keras.layers.Dense(self._ae_l2_fmaps, 'relu')(e)
        d = tf.keras.layers.Dense(self._ae_l1_fmaps, 'relu')(d)
        output = tf.keras.layers.Dense(self._fvec_size, 'relu')(d)

        self._AE = tf.keras.Model(input_vec, output)

    def _grain_to_tensor(self, grain_in: PIL.Image.Image):
        """Processes a single grain for use in our model

        Params
        ------
        grain_in : Image.Image
            An image from the rover's "camera" that needs to be preprocessed

        Return
        ------
        A grain that has been reprocessed and is now a tensor
        """

        grain = tf.keras.preprocessing.image.img_to_array(grain_in)
        grain = tf.image.per_image_standardization(grain) # Transform images to zero mean and unit variance
        grain = tf.image.resize(grain, (self._image_width, self._image_width)) # Resize to CNN base input size
        return grain

    def add_grains(self, grains: List[PIL.Image.Image]):
        """Add new grains to memory

        Params:
            grains: List[PIL.Image.Image]
                List of new grains

        Returns:
            List of novelty for new grains
        """

        assert len(grains) == 4 # Currently, we only allow 4 grains
        nov_list = []

        for g in grains:
            gtf = self._grain_to_tensor(g)
            gtf = tf.reshape(gtf, (1, gtf.shape[0], gtf.shape[1], gtf.shape[2]))
            fvec = self._CNN(gtf)
            pred_fvec = self._AE(fvec)
            nov = self.novelty_function(fvec, pred_fvec).numpy()
            nov_list.append(nov)
            self.memory.push(Experience(nov, fvec.numpy(), g))
            
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
        
        gradients = tape.gradients(loss, self._AE.trainable_variables)
        self._AE_opt.apply_gradients(zip(gradients, self._AE.trainable_variables))
        return loss

    def learn_grains(self):
        """Train the AE to learn new features from memory"""

        memList = self.memory.memList()
        fvecs = list(map(lambda e: e.featureVector, memList))
        dataset = tf.data.Dataset.from_tensor_slices(fvecs).shuffle(self._batch_size).repeat().batch(self._batch_size)
        dataset = iter(dataset)

        num_batches = math.ceil(len(memList()) / self._batch_size)
        cur_avg_loss = float('inf')

        while cur_avg_loss >= self.nov_thresh:
            cur_avg_loss = 0
            for batch in range(num_batches):
                data = dataset.next()
                loss = self._train_step_ae(data).numpy()
                cur_avg_loss += (loss/num_batches)

    # def _prep_grains(self, grains: List[PIL.Image.Image], batch_size: int):
    #     """Prepares the grains used for finding directions. We have a specific method for this because the batch will always be zero
    #     and because I haven't written anything else yet

    #     Params
    #     ------
    #     grains : PIL.Image.Images
    #         The images that need to be processed
    #     batch_size : int
    #         The size of the batches that will be prepared

    #     Return
    #     ------
    #     A dataset split into a batch size of four
    #     """
    #     assert batch_size != None
    #     prep_grains = list(map(self._grain_to_tensor, grains))
    #     return tf.data.Dataset.from_tensor_slices(prep_grains).shuffle(batch_size).repeat().batch(batch_size)

    # def _run_through_main_model(self, grains: List[PIL.Image.Image]):
    #     """This function is just an example of how to send grains through the entiere main model and get out
    #     pairs of feature/reconstruction vectors

    #     Params
    #     ------
    #     grains : Image.Images
    #         The images that will be sent through the model

    #     Return
    #     ------
    #     A list of the feature vector/reconstruced vector pairs from the grains. They are returned in the same order that they were
    #     passed in
    #     """

    #     # Prepare the images for the CNN
    #     vect_images = self._prep_grains(grains, 4)
    #     # Generate feature vectors
    #     feature_vects = self._CNN(list(iter(vect_images))[0])
    #     # Run the vectors through the auto-encoder to get reconstruction vectors
    #     reconstructed = self._AE(feature_vects)
    #     # Return the pairs of feature/reconstructed
    #     return list(zip(feature_vects, reconstructed))

    # def eval_novelty(self, grains: List[PIL.Image.Image]):
    #     # This is because we haven't decided to allow different numbers of grains yet
    #     assert len(grains) == 4
    #     pairs = self._run_through_main_model(grains)
    #     loss = [Experience(l1_loss(*pairs[x]), pairs[x][0], grains[x])
    #             for x in range(len(grains))]
    #     return loss

    # def learn_grains(self, exper: List[Experience], memory: Memory = None):

    #     # Const for batch size
    #     batch_size = 4

    #     # Create new list so we can reference old one later
    #     newList = exper
    #     if memory != None:
    #         newList += memory.memList()

    #     # Extract feature vectors and create batches
    #     feature_vects = list(map(lambda g: g.featureVector, newList))
    #     batches = tf.data.Dataset.from_tensor_slices(
    #         feature_vects).batch(batch_size)

    #     # create an object to hold the loss so we can check it as we go
    #     current_loss = 1000
    #     last_loss = 1000  # some large number
    #     diff = self.nov_thresh

    #     num_batches = math.ceil(len(feature_vects)/batch_size)

    #     # Here is where we do the actual training
    #     while diff >= self.nov_thresh:
    #         last_loss = current_loss
    #         current_loss = 0
    #         for batch in iter(batches):
    #             loss_value, grads = self._AE_grad(batch)
    #             self._AE_optimizer.apply_gradients(
    #                 zip(grads, self._AE.trainable_variables))
    #             current_loss += loss_value

    #         current_loss /= num_batches
    #         diff = last_loss - current_loss
    #         print(diff)

    #     # Put grains in long-term memory
    #     for e in exper:
    #         memory.push(e)

if __name__ == "__main__":
    im = PIL.Image.open('data/x.jpg')
    brain = Brain(0.0005, 'MSE')

    # m = Memory()
    # testing_brain = Brain(0.005, m)

    # lst = testing_brain.eval_novelty([im, im, im, im])
    # print(list(map(lambda x : x.novelty, lst)))

    # for e in lst:
    #     m.push(e)

    # testing_brain.learn_grains(lst, m)

    # lst2 = testing_brain.eval_novelty([im, im, im, im])
    # print(list(map(lambda x : x.novelty, lst2)))
