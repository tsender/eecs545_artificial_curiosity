import heapq
from typing import Tuple, List, Generator
from experience import Experience
import pprint
import abc
from PIL import Image

class BaseMemory(metaclass=abc.ABCMeta):
    """
    Base memory class for the agent's brain.

    Methods

    `__init__(maxLength: int = 32)`  
        Initializes the memory unit with a default capacity of 32 Experiences
    `push(data: Experience)`  
        Adds an Experience to the memory unit.
    `as_list() -> List[Experience]`  
        Returns a list of Experience instances
    """

    def __init__(self, max_length: int = 32):
        """
        Args:
            maxLength : int  
                The maximum number of experiences(Experience) that the memory unit can contain
        """
        self._memory = []
        self._max_length = max_length

    @abc.abstractmethod
    def push(self, data: Experience):
        """ Add an experience to memory
        
        Args
            data : Experience  
                An experience to add
        """
        pass

    def as_list(self) -> List[Experience]:
        """ Returns a copy of the current memory

        Returns
            A list of Experience objects
        """
        return self._memory.copy()

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return pprint.pformat(vars(self))
