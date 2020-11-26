from artificial_curiosity_types import Artificial_Curiosity_Types as act
from typing import Tuple
import abc
from brain import Brain
from map import Map
import random

RATE = 1

class Motivation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_from_position(self, position: Tuple[int]):
        pass

    def _generate_positions(self, position):
        lt = (position[0]-RATE, position[1]+RATE)
        lb = (position[0]-RATE, position[1]-RATE)
        rt = (position[0]+RATE, position[1]+RATE)
        rb = (position[0]+RATE, position[1]-RATE)

        return [lt, lb, rt, rb]


class Curiosity(Motivation):
    def __init__(self, map: Map):
        self._brain = Brain(nov_thresh=0.25, novelty_loss_type='mse')
        self._map = map

    def get_from_position(self, position: Tuple[int]):
        grains = self._map.get_fov(position)
        novelty = self._brain.evaluate_novelty(grains)
        new_positions = self._generate_positions(position)
        position_filter = self._map.clean_directions(new_positions)
        index = self._max_pos(novelty, position_filter)
        return new_positions[index]

    def _max_pos(self, lst=[], filter=[]):
        assert lst is not [] and filter is not []
        pos = 0
        max = min(lst)

        # TODO: There might be some weird edge cases here
        # because we could have 0 available positions (synthetically)
        for i in range(len(lst)):
            if filter[i] and lst[i] > max:
                max = lst[i]
                pos = i

        return pos

class Brownian(Motivation):
    def __init__(self, map: Map):
        self._map = map

    def get_from_position(self, position: Tuple[int]):
        new_positions = self._generate_positions(position)
        position_filter = self._map.clean_directions(new_positions)
        index = random.choice([i for i in range(len(position_filter)) if position_filter[i]])

        return new_positions[index]

class Linear(Motivation):
    def __init__(self, map: Map):
        self._map = map
        self.direction = random.randint(0,3)

    def get_from_position(self, position: Tuple[int]):
        new_positions = self._generate_positions(position)
        position_filter = self._map.clean_directions(new_positions)
        
        if position_filter[self.direction]:
            return new_positions[self.direction]
        else:
            if self.direction == 0:
                self.direction = 2
            elif self.direction == 1:    
                self.direction = 3
            elif self.direction == 2:   
                self.direection = 0
            else:   
                self.direction = 2 
            
            return self.get_from_position(position)


if __name__ == "__main__":
    map = Map('data/x.jpg', 30, 2)

    print(Curiosity(map=map).get_from_position((30, 30)))
    print(Brownian(map=map).get_from_position((30, 30)))
    print(Linear(map=map).get_from_position((30, 30)))
