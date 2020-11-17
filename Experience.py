from typing import Tuple, List, Generator
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as act

class Experience:
    # A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory

    def __init__(self, nov: act.Novelty, fVect: List[float], grn: act.Image):
        self.novelty = nov
        self.featureVector = fVect
        self.grain = grn

    def __lt__(self, other):
        if(isinstance(other, Experience)):
            return self.novelty < other.novelty
        else:
            return self.novelty < other

    def __le__(self, other):
        return not self.__gt__(other)

    def __gt__(self, other):
        if(isinstance(other, Experience)):
            return self.novelty > other.novelty
        else:
            return self.novelty > other

    def __ge__(self, other):
        return not self.__lt__(other)

    def __eq__(self, other):
        if(isinstance(other, Experience)):
            return self.novelty == other.novelty
        else:
            return self.novelty == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "novelty: {0}\nfeatureVector {1}\ngrain: {2}".format(self.novelty, self.featureVector, self.grain)

