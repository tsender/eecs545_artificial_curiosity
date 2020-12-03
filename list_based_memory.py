from typing import Tuple, List, Generator
import pprint
import heapq
from PIL import Image

from base_memory import BaseMemory
from experience import Experience

class ListBasedMemory(BaseMemory):
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

if __name__ == "__main__":
    print = pprint.PrettyPrinter(indent=4).pprint
    m = ListBasedMemory(5)
    for i in range(5):
        m.push(Experience(i, None))

    print(m)
    m.push(Experience(6, None))
    print(m)

    for i in m.as_list():
        print(i.novelty)
