from typing import Tuple, List, Generator
import pprint
import heapq
from PIL import Image

from base_memory import BaseMemory
from experience import Experience

class PriorityBasedMemory(BaseMemory):
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

if __name__ == "__main__":
    print = pprint.PrettyPrinter(indent=4).pprint
    m = PriorityBasedMemory(5)
    for i in range(5):
        m.push(Experience(i, None))

    print(m)
    m.push(Experience(6, None))
    print(m)

    for i in m.as_list():
        print(i.novelty)
