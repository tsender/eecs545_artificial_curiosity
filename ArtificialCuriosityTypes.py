from typing import Tuple, List, Generator
from types import SimpleNamespace
from PIL import Image

# New types to simplify typing for this project
ArtificialCuriosityTypes = SimpleNamespace()

def addType(**args):
    """Makes it easier to add types that reference other custom types
        Parameters
        __________
        **args : typeing.Type
            Can add any type by passing <name> = <type>, where <type> is a type from the typing library
        
        Returns
        _______
        None
        """
    ArtificialCuriosityTypes.__dict__.update(args)

# A type that simply notates where novelty should be used
addType(Novelty=float)

# A type to represent a feature vector
addType(FeatureVector=List[float])

# A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory

addType(Grain = Image.Image)
