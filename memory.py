import heapq
from typing import Tuple, List, Generator
from artificial_curiosity_types import Artificial_Curiosity_Types as act
from experience import Experience
import pprint
from PIL import Image

class Memory:
    """
    This class abstracts away the specific implementation of an autonomous agent's memory unit

    Attributes

    None


    Methods

    `__init__(maxLength: int = 32)`  
        Initializes the memory unit with a default capacity of 32 Experiences
    `push(data: Experience)`  
        Adds an Experience to the memory unit. If the memory is full, it forgets the Experience that had the smallest act.Novelty
    `memList() -> List[Experience]`  
        Returns a list of Experience instances
    """

    def __init__(self, maxLength: int = 32):
        """
        Parameters
        ---------
        maxLength : int  
            The maximum number of experiences(Experience) that the memory unit can contain
        
        Returns
        -------
        Memory
        """
        self._heap: List[Experience] = []
        self.maxLength: int = maxLength

    def push(self, data: Experience):
        """
        ### Parameters

        > data : Experience  
        >    > Adds an experience (Experience) to memory. Once full, experiences that are less novel (lower values of act.Novelty) will be forgotten as new experiences are added

        Returns

        None
        """
        if(len(self._heap) < self.maxLength):
            heapq.heappush(self._heap, data)
        # Do nothing if less than the smallest element because it would not be interesting enough to remember
        elif(data > self._heap[0]):
            heapq.heappushpop(self._heap, data)

    def memList(self) -> List[Experience]:
        """
        Parameters

        None

        Returns

        List[Experience]
            A list of Experience objects
        """
        return self._heap

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return pprint.pformat(vars(self))

if __name__ == "__main__":
    print = pprint.PrettyPrinter(indent=4).pprint
    m = Memory(5)
    for i in range(5):
        m.push(Experience(i, None, None))

    print(m)
    m.push(Experience(6, None, None))
    print(m)

    for i in m.memList():
        print(i.novelty)
