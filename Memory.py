import heapq
from typing import Tuple, List, Generator
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as ac

class Memory:
    """
    Thsi class abstracts away the specific implementation of an autonomous agent's memory unit

    

    Attributes
    ----------
    None


    Methods
    -------
    __init__(maxLength: int = 30)
        Initializes the memory unit with a default capacity of 30 ac.Experience
    push(data: Experience)
        Adds an ac.Experience to the memory unit. If the memory is full, it forgets the ac.Experience that had the gratest ac.Novelty
    memIter() -> Generator
        Creates an iterator that can be used to iterate over ac.Experience instances
    """

    def __init__(self, maxLength: int = 30):
        """
        Parameters
        __________
        maxLength : int
            The maximum number of experiences(ac.Experience) that the memory unit can contain
        
        Returns
        _______
        Memory
        """
        self.heap: List[ac.Experience] = []
        self.maxLength: int = maxLength

    def push(self, data: ac.Experience):
        """
        Parameters
        __________
        data : ac.Experience
            Adds an experience (ac.Experience) to memory. Once full, experiences that are less novel (lower values of ac.Novelty) will be forgotten as new experiences are added

        Returns
        _______
        None
        """
        if(len(self.heap) < self.maxLength):
            heapq.heappush(self.heap, data)
        # Do nothing if less than the smallest element because it would not be interesting enough to remember
        elif(data[0] > self.heap[0][0]):
            heapq.heappushpop(self.heap, data)

    def memIter(self) -> Generator:
        """
        Parameters
        __________
        None

        Returns
        _______
        Generator
            An iterator that operates over all experiences (ac.experience) in memory
        """
        return iter(self.heap)
