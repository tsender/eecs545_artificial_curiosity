
# Need this because our naming conventions are messing with the default map function
map_function = map

from typing import List, Tuple
import numpy as np

from map import Map

def load_from_map(map: Map, positions: List[Tuple[int]]):
    """
    Grabs images at a list of coordinates, converts them to a numpy array, and returns them

    Params
    ------

    map: Map
        The map that you want to get the images from

    positions: List[Tuple[int]]
        A list of all of the points that will be sampled and turned into an numpy array

    Returns
    -------

    A numpy array of dimension (<number_of_images>, <height_of_images>, <width_of_images>
    """
    return np.asarray(list(map_function(lambda x: np.array(map.full_view(x)), positions)))


def avg_pixelwise_var(images_seen: np.int16):
    """
    Computes the variance for every pixel p across all images, resulting in a matrix holding
    the variance for eack pixel p, then calculates the average of that variance across all
    pixels. This allows us to compensate for different fov sizes.
    Note: images are normalized to [-1,1] before calculations

    Params
    ------

    images_seen
        A numpy matrix holding numpy versions of all of our images

    Returns
    -------

    The average pixelwise variation across all images, as a float
    """

    # Computes the variance
    images = (images_seen.astype(np.float32) - 127.5) / 127.5 # Normalize to [-1,1]
    variance_matrix = np.var(images, 0)
    # Returns the average of that variance
    return(np.sum(variance_matrix)/variance_matrix.size)


if __name__ == "__main__":
    m = Map("./data/mars.png", 64, 2)

    images = load_from_map(m, [(100, 100),(200,200),(300,300)])
    print(avg_pixelwise_var(images))
