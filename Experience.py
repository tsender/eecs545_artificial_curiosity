from typing import Tuple, List, Generator
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as act

class Experience:
    """
    A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory and the Brain

    

    Attributes
    ----------
    self.novelty : act.Novelty
        A float that represents the movelty of the Experience
    self.featureVector : List[float]
        A vector that holds the different features that represent this memory
    self.grain : act.Image
        An image that will show us what the machine remembers


    Methods
    -------
    __init__(nov: act.Novelty, fVect: List[float], grn: act.Image)
        Initializes the Experience with the given novelty, feature vector, and image
    __lt__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __le__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __gt__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __ge__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __eq__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __ne__(other)
        Compares against the novelty. Works for scalars and other instances of Experience
    __str__()
        Returns a string representation of Experience
    """

    def __init__(self, nov: act.Novelty, fVect: List[float], grn: act.Image):
        """
        Parameters
        __________
        nov : act.Novelty
            The measure of novelty, expressed as a float
        fVect : List[float]
            A feature vector expressing the grain (image)
        grn: act.Image
            A grain (image) to be remembered. This exists so we can reference it later
        
        Returns
        _______
        Experience
        """

        self.novelty = nov
        self.featureVector = fVect
        self.grain = grn

    def __lt__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return self.novelty < other

    def __le__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return not self.__gt__(other)

    def __gt__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return self.novelty > other

    def __ge__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return not self.__lt__(other)

    def __eq__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return self.novelty == other

    def __ne__(self, other):
        """
        Parameters
        __________
        other : any
        
        Returns
        _______
        bool
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        Parameters
        __________
        None
        
        Returns
        _______
        string
        """
        return "novelty: {0}\nfeatureVector {1}\ngrain: {2}".format(self.novelty, self.featureVector, self.grain)

