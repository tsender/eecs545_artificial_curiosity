import heapq
from typing import Tuple, List, Generator
import pprint
import abc
from PIL import Image

from experience import Experience

class BaseMemory(metaclass=abc.ABCMeta):
    """
    Base memory class for the agent's brain.

    Methods

    `__init__(maxLength: int = 64)`  
        Initializes the memory unit with a default capacity of 64 Experiences
    `push(data: Experience)`  
        Adds an Experience to the memory unit.
    `as_list() -> List[Experience]`  
        Returns a list of Experience instances
    """

    def __init__(self, max_length: int = 64):
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

    @abc.abstractmethod
    def get_name(self):
        """Returns the name of the memory object.
        
        Returns
            The name of the memory object as a string
        """
        pass

    def as_list(self) -> List[Experience]:
        """ Returns a copy of the current memory

        Returns
            A list of Experience objects
        """
        return self._memory.copy()

    def get_max_length(self):
        return self._max_length

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return pprint.pformat(vars(self))

class PriorityMemory(BaseMemory):
    """
    Memory class that uses a fixed-length priority queue to store experiences based on their novelty.
    Low novelty corresponds to higher priority (also makes it easier to remove the experience).
    """

    def __init__(self, max_length: int = 64):
        super().__init__(max_length)

    def push(self, data: Experience):
        """ Add an experience to memory
        
        Args
            data : Experience  
                An experience to add. If full, experiences that are less novel are removed (forgotten).
        """

        if(len(self._memory) < self._max_length):
            heapq.heappush(self._memory, data)
        elif(data > self._memory[0]):
            heapq.heappushpop(self._memory, data) # New data must be more interesting than the least interesting experience

    def get_name(self):
        """Returns the full descriptive name of the memory object.
        
        Returns
            The name of the memory object as a string
        """
        return "PriorityMem_Mem" + str(self._max_length)


class CircularMemory(BaseMemory):
    """
    Memory class that uses a simple fixed-length list to store the latest experiences.
    """

    def __init__(self, max_length: int = 64):
        super().__init__(max_length)

    def push(self, data: Experience):
        """ Add an experience to memory
        
        Args
            data : Experience  
                An experience to add. If full, remove oldest experience and add new experience.
        """

        self._memory.append(data)
        if(len(self._memory) > self._max_length):
            self._memory.pop(0)

    def get_name(self):
        """Returns the full descriptive name of the memory object.
        
        Returns
            The name of the memory object as a string
        """
        return "CircMem_Mem" + str(self._max_length)

if __name__ == "__main__":
    print("Priority Memory")
    print = pprint.PrettyPrinter(indent=4).pprint
    m = PriorityMemory(5)
    for i in range(5):
        m.push(Experience(i, None))

    print(m)
    m.push(Experience(6, None))
    print(m)

    for i in m.as_list():
        print(i.novelty)

    print("Circular Memory")
    m = CircularMemory(5)
    for i in range(5):
        m.push(Experience(i, None))

    print(m)
    m.push(Experience(6, None))
    print(m)

    for i in m.as_list():
        print(i.novelty)

    print(m.get_name())