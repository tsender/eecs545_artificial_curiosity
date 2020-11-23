import heapq
from typing import Tuple, List, Generator
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as act
from Experience import Experience
import pprint

class Memory:
    """
    This class abstracts away the specific implementation of an autonomous agent's memory unit

    Attributes

    None


    Methods

    `__init__(maxLength: int = 30)`  
        Initializes the memory unit with a default capacity of 30 Experience
    `push(data: Experience)`  
        Adds an Experience to the memory unit. If the memory is full, it forgets the Experience that had the gratest act.Novelty
    `memIter() -> Generator`  
        Creates an iterator that can be used to iterate over Experience instances
    """

    def __init__(self, maxLength: int = 30):
        """
        ### Parameters
        
        > maxLength : int  
        >    > The maximum number of experiences(Experience) that the memory unit can contain
        
        ### Returns

        > Memory
        """
        self.heap: List[Experience] = []
        self.maxLength: int = maxLength

    def push(self, data: Experience):
        """
        ### Parameters

        > data : Experience  
        >    > Adds an experience (Experience) to memory. Once full, experiences that are less novel (lower values of act.Novelty) will be forgotten as new experiences are added

        Returns

        None
        """
        if(len(self.heap) < self.maxLength):
            heapq.heappush(self.heap, data)
        # Do nothing if less than the smallest element because it would not be interesting enough to remember
        elif(data > self.heap[0]):
            heapq.heappushpop(self.heap, data)

    def memIter(self) -> Generator:
        """
        ### Parameters

        > None

        ### Returns

        > Generator
        >    > An iterator that operates over all experiences (Experience) in memory
        """
        return iter(self.heap)

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

    for i in m.memIter():
        print(i.novelty)
