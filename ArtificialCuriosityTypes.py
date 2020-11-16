from typing import Tuple, List, Generator
from types import SimpleNamespace

# New types to simplify typing for this project
ArtificialCuriosityTypes = SimpleNamespace()


def addType(**args):
    ArtificialCuriosityTypes.__dict__.update(args)


# A type that represents an image
addType(Image = List[List[float]])

# # A type that simply notates where novelty should be used
addType(Novelty=float)

# # A type to represent a feature vector
addType(FeatureVector=List[float])

# # A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory
addType(
    Experience = Tuple[
        ArtificialCuriosityTypes.Novelty,
        ArtificialCuriosityTypes.FeatureVector,
        ArtificialCuriosityTypes.Image
    ])
