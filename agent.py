from artificial_curiosity_types import Artificial_Curiosity_Types as act
from typing import Tuple
import abc
from brain import Brain
from map import Map


class Motivation(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_from_position(self, position: Tuple[int]):
        pass


class Curiosity(Motivation):
    def __init__(self, map: Map):
        self._brain = Brain(nov_thresh=0.25, novelty_loss_type='mse')
        self._map = map

    def get_from_position(self, position: Tuple[int]):
        grains = self._map.get_fov(position)
        novelty = self._brain.evaluate_novelty(grains)
        positions = self._generate_positions()
        can_move = self._map.clean_directions()
        direction = self._max_pos(novelty)
        return self._translate_position(direction)

    def _max_pos(self, lst=[]):
        assert lst is not []
        pos = 0
        max = lst[0]

        for i in range(len(lst)):
            if lst[i] > max:
                max = lst[i]
                pos = i

        return pos    

    def _generate_positions(self):
        



class Generic_Agent:

    def __init__(self, motivation: Motivation, position: Tuple[int] = (0, 0)):
        assert motivation is not None

        self._motivation = motivation
        self.position = position
        self.history = [position]

    def step(self):
        new_position = self._motivation.get_from_position(self.position)
        self.history.append(new_position)
        self.position = new_position

