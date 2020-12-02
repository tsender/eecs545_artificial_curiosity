from typing import Tuple, List, Generator
import pprint
from PIL import Image
import numpy as np

class Experience:
    """
    A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory and the Brain

    Attributes
    ----------
    self.novelty : float
        A float that represents the movelty of the Experience
    self.grain : Image.Image
        An image that will show us what the machine remembers


    Methods
    -------
    __init__(nov: float, grn: Image.Image)
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

    def __init__(self, nov: float, grn: Image.Image):
        """
        Parameters
        __________
        nov : float
            The measure of novelty, expressed as a float
        grn: Image.Image
            A grain (image) to be remembered. This exists so we can reference it later
        
        Returns
        _______
        Experience
        """

        self.novelty = nov
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
        return str(vars(self))

    def __repr__(self):
        return pprint.pformat(vars(self), indent=1, width=1)

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)

    print(Experience(0, None) <  Experience(1, None))
    print(Experience(0, None) <= Experience(1, None))
    print(Experience(0, None) >  Experience(1, None))
    print(Experience(0, None) >= Experience(1, None))
    print(Experience(0, None) == Experience(1, None))
    print(Experience(0, None) != Experience(1, None))
    pp.pprint(Experience(0, None))
