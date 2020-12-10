# Need this because the name map is taken by the actual map
lst_map = map

from typing import Tuple, List
import abc
import os
import csv
import random
import numpy as np
from PIL import Image

from brain import Brain
from map import Map
import evaluate

# Abstract Class
class Motivation(metaclass=abc.ABCMeta):
    """This is an abstract class that represents the decision making process (or motivation) behind an agent"""

    @abc.abstractmethod
    def get_from_position(self, position: Tuple[int]):
        """This will get the agents next position based on the current position passed to it"""
        pass

    def _generate_positions(self, position: Tuple[int]):
        """This generates the possible positions based on the current position. This will be filtered by other methods later on"""

        # TODO: These need to be confirmed by Aravind because I'm not sure that I'm handing the directions properly
        # It's a little confusing because the origin is in the top left because that's how PIL handles images

        # Left top
        lt = (position[0]-self.rate, position[1]-self.rate)
        # Left bottom
        lb = (position[0]-self.rate, position[1]+self.rate)
        # Right top
        rt = (position[0]+self.rate, position[1]-self.rate)
        # Right bottom
        rb = (position[0]+self.rate, position[1]+self.rate)

        return [
            [lt, rt],
            [lb, rb]
        ]


class Curiosity(Motivation):
    """This class extends Motivation and creates a curious motivation for an agent"""

    def __init__(self, map: Map, brain: Brain, rate: int = 1, prob: Tuple[float] = (1.0, 0.0)):
        # This assigns a map to the agent, which is where they will get their directions from
        self.map = map
        # Creates a brain for the agent with some parameters
        # TODO: These are not optimal, they have been set for testing purposes
        self._brain = brain
        self.rate = rate
        self.perceived_path_novelty_history = []
        self.grain_novelty_history = []
        self._prob = prob # Probability that we choose the 1st and 2nd best options at each step

    def get_map(self):
        return self.map

    def get_from_position(self, position: Tuple[int]):
        """Implements the abstract method from Motivation. Gets the next position from the current position"""

        # Gets images from the map
        grains = self.map.get_fov(position)
        # Finds the novelty of those images
        novelty, _ = self._brain.evaluate_grains(grains)

        # Save novelty history
        novelty_expanded = novelty[0] + novelty[1]
        self.grain_novelty_history.append(novelty_expanded)

        # Finds the potential new positions from the current position
        new_positions = self._generate_positions(position)
        # Makes sure that the agent doesn't go outside of the map
        # Creates a list of booleans to act as a filter for the list of positions
        position_filter = self.map.clean_directions(new_positions)
        # Finds the index with the greatest novelty that is also a valid position
        x,y = self._max_pos_stochastic(novelty, position_filter)
        self.perceived_path_novelty_history.append(novelty[y][x])

        # Add the grains to memory
        self._brain.add_grains(grains)
        # Train on the grains in memory
        self._brain.learn_grains()
        # Return the chosen position
        return new_positions[y][x]

    def _max_pos_stochastic(self, novelty: List[List[int]]=[], filter: List[List[bool]]=[]):
        """Finds the position with the maximum novelty given a filter list (which labels positions as valid or invalid),
        but use a probability to indicate what fraction of the time you pick the best or second best position.
        """
        assert novelty is not [] and filter is not []
        assert len(novelty) == len(filter)
        assert len(novelty[0]) == len(filter[0])

        num_rows = 0
        for row in filter:
            num_rows += sum(row)
        pos_filter_array = np.zeros((num_rows, 3))

        num_valid = 0
        for y in range(len(novelty)):
            for x in range(len(novelty[y])):
                if filter[y][x]:
                    pos_filter_array[num_valid,:] = np.array([novelty[y][x], x, y])
                    num_valid += 1

        # Initialize x,y to the best position
        idx = np.argsort(pos_filter_array, axis=0)[:,0]
        idx = np.flip(idx)
        best_row = idx[0]
        x = pos_filter_array[best_row, 1]
        y = pos_filter_array[best_row, 2]

        i = 0
        lower = 0
        upper = self._prob[0]
        p = random.random()
        for val in self._prob:
            if lower <= p and p < upper:
                row = idx[i]
                x = pos_filter_array[row, 1]
                y = pos_filter_array[row, 2]
                break
            i += 1
            lower += val
            upper += val
            if i == num_valid:
                break

        return (int(x), int(y))


    def _max_pos(self, lst:List[List[int]]=[], filter:List[List[bool]]=[]):
        """Finds the position with the maximum novelty given a filter list (which labels positions as valid or invalid)"""

        # Make sure that the lists are not empty
        assert lst is not [] and filter is not []
        # Initializes the positon to 0.5 so we can catch errors.We can't use -1 to catch errors because -1 will just be the last index
        pos = 0.5
        max = min(list(lst_map(min, lst)))-1

        # TODO: There might be some weird edge cases here because we could have 0 available positions
        #  This would not be observed in the real world though
        for y in range(len(lst)):
            for x in range(len(lst[y])):
                if filter[y][x] and lst[y][x] > max:
                    # Update the position
                    max = lst[y][x]
                    pos = (x,y)

        # Return the position that has the highest novelty
        return pos

    def get_reconstruction_snapshot(self, position: Tuple[int]):
        """Saves the full-view and its reconstruction from the current position"""
        fullview = self.map.full_view(position)
        fov_grains = self.map.get_fov(position)
        _, pred_grains = self._brain.evaluate_grains(fov_grains)

        # Stitch together predicted grains
        fov = self.map.fov
        pred_fullview = Image.new('L', (2*fov, 2*fov))
        pred_fullview.paste(im=pred_grains[0][0], box=(0, 0))
        pred_fullview.paste(im=pred_grains[0][1], box=(fov, 0))
        pred_fullview.paste(im=pred_grains[1][0], box=(0, fov))
        pred_fullview.paste(im=pred_grains[1][1], box=(fov, fov))

        return fullview, pred_fullview

    def save_data(self, data_dir: str):
        # Save grain novelty history
        with open(os.path.join(data_dir, "grain_novelty.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.grain_novelty_history)

        # Save grain novelty history
        with open(os.path.join(data_dir, "perceived_path_novelty.csv"), "w") as f:
            writer = csv.writer(f)
            for val in self.perceived_path_novelty_history:
                writer.writerow([val])

    def __str__(self):
        return "Curiosity_Prob" + str(self._prob[0]) + "_" + self._brain.get_name()


class Random(Motivation):
    """This class extends Motivation, and randomly selects a position based on what is available"""

    def __init__(self, map: Map, rate: int = 1):
        # Saves the given map
        self.map = map
        self.rate = rate
        self.data_dir = None

    def get_map(self):
        return self.map

    def get_from_position(self, position: Tuple[int]):
        """Implements the abstract method from Motivation. Gets the next position from the current position"""

        # Gets a list of new positions
        new_positions = self._generate_positions(position)
        # Creates a filter for the ones that are in the map or out of it
        position_filter = self.map.clean_directions(new_positions)

        #  In this case it's easier to flatten them and chose than to work with a 2d array
        flattened_positions = (new_positions[0] + new_positions[1])
        flattened_filter = (position_filter[0] + position_filter[1])

        if True not in flattened_filter:
            print("Random Agent: At pos ", position, " has no valid moves")
            print("No valid moves ", self._generate_positions(position))

        # Chooses a random index from the position list random position that is allowed by the filter
        # TODO: We can probably modify this so that the last two lines are combined, selecting a position itself
        # instead of an index for a position and then getting the position at the index
        index = random.choice([i for i in range(len(flattened_filter)) if flattened_filter[i]])

        # Return that 
        return flattened_positions[index]

    def save_data(self, data_dir: str):
        pass

    def get_reconstruction_snapshot(self, position: Tuple[int]):
        return None, None

    def __str__(self):
        return "Random"

class Linear(Motivation):
    """This class extends Motivation, and is designed to move on a linear path"""

    def __init__(self, map: Map, rate: int = 1):
        # Saves the map
        self.map = map
        # Chooses a random direction to move in
        self.direction = (random.randint(0, 1), random.randint(0, 1))
        self.rate = rate

    def get_map(self):
        return self.map

    def get_from_position(self, position: Tuple[int]):
        """Implements the abstract method from Motivation. Gets the next position from the current position"""

        # Find the possible new positions
        new_positions = self._generate_positions(position)
        # Creates a filter for the ones that are in the map or out of it
        position_filter = self.map.clean_directions(new_positions)
        
        # Cycles through all of the options until a valid position is found
        while not position_filter[self.direction[1]][self.direction[0]]:
            # I'll explain on one and the rest should be self explanatory

            # Assume we're going diagonally in a specific direction and get stopped by something.
            # We can be stuck in one of two ways: we have hit a wall and two positions are no longer
            # available, or we've perfectly hit a corner and three positions are no longer available
            # It's easier to handle the case with three positions as a combination (or two iterations) of 
            # the case with two positions, which is why we have the while loop (it just keeps reapplying
            # the case with two positions until a valid position is found).

            # Since we want our agent to act like it bounces off the barriers, we have to choose a direction
            # We don't want to move back along the path that we came from, because that's ridiculous. So, we
            # need to find the two directions that are orthogonal to the given direction. Given that our dimensions
            # are shaped as [[0,2],[1,3]], we know that directions 1 and 2 are orthogonal to 0. However, depending
            # on the way that we are aligned to an edge, one of these positions will be blocked (e.g. if we are
            # blocked from moving right-diagonally up, it is either because we have hit the ceiling, which eliminates
            # left-diagonally up, or we have hit the right wall, which eliminates right-diagonally down). We can do
            # this logic with if statements, but it is cleaner to simply choose a random direction and have the
            # algorithm sort it out, since if the random direction is wrong, it will perform the checks again. This
            # could theoretically run in an infinite loop, but this is unlikely and modern CPUs make the time
            # difference trivial            

            if(self.direction[0] == 0 and self.direction[1] == 0):
                self.direction = random.choice(((0,1), (1,0)))
            elif(self.direction[0] == 0 and self.direction[1] == 1):
                self.direction = random.choice(((0,0), (1,1)))
            elif(self.direction[0] == 1 and self.direction[1] == 0):
                self.direction = random.choice(((0,0), (1,1)))
            else:
                self.direction = random.choice(((0, 1), (1, 0)))

        # Return the chosen position
        return new_positions[self.direction[1]][self.direction[0]]

    def save_data(self, data_dir: str):
        pass

    def get_reconstruction_snapshot(self, position: Tuple[int]):
        return None, None
    
    def __str__(self):
        return "Linear"


class Agent:
    """This abtracts the motivation away so that we can iterate over them later"""
    def __init__(self, motivation: Motivation, position: Tuple[int] = (0, 0), steps_bw_snapshot: int = 50):
        # Make sure that it was passed valid values
        assert motivation is not None

        # Save the motivation
        self._motivation = motivation
        # Save the current position
        self.position = position
        # Save the position to its history (which we will use to reconstruct the path later)
        self.history = [position]
        self.data_dir = self.__str__()
        self.step_num = 0
        self.steps_bw_snapshot = steps_bw_snapshot

    def set_data_dir(self, data_dir: str):
        """Set a new data directory"""
        self.data_dir = data_dir

    def step(self):
        """This performs a simple step for the agent, moving it from one position to the next"""

        if self.step_num == 0 or (self.step_num % self.steps_bw_snapshot == 0):
            self.save_reconstruction_snapshot()

        # Get the new position based on the motivation
        new_position = self._motivation.get_from_position(self.position)
        # Add the new position to its history
        self.history.append(new_position)
        # Update its position based on the new position
        self.position = new_position
        self.step_num += 1

    def save_reconstruction_snapshot(self):
        """Saves the full-view and its reconstruction from the current position"""
        fullview, pred_fullview = self._motivation.get_reconstruction_snapshot(self.position)

        if fullview is None:
            return

        dir = os.path.join(self.data_dir, "snapshots")
        os.makedirs(dir, exist_ok=True)

        orig_name = os.path.join(dir, "step" + str(self.step_num) + "_orig.png")
        pred_name = os.path.join(dir, "step" + str(self.step_num) + "_pred.png")
        fullview.save(orig_name, "PNG")
        pred_fullview.save(pred_name, "PNG")

    def get_map(self):
        return self._motivation.get_map()

    def get_path_novelty(self):
        """Return average path variance. Note, images are converted to the range [-1,1]"""
        images = evaluate.load_from_map(self.get_map(), self.history)
        return evaluate.avg_pixelwise_var(images)

    def save_data(self):
        """Save agent's data to a folder"""
        os.makedirs(self.data_dir, exist_ok=True)
        # Save the path coordinates to a file
        with open(os.path.join(self.data_dir, "path_record.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.history) # x,y format

        # Save avg path variance
        with open(os.path.join(self.data_dir, "avg_path_variance.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow([self.get_path_novelty()])

        # Save any additional data relevant to the motivation (applies to Curiosity)
        self._motivation.save_data(self.data_dir)

    def __str__(self):
        return F"{self._motivation}_Agent_Pos_{self.history[0][0]}_{self.history[0][1]}"


if __name__ == "__main__":
    from memory import PriorityBasedMemory, ListBasedMemory
    fov = 64 # Allowed FOVs = {32, 64, 128}
    map = Map('data/mars.png', fov, 2)

    brain = Brain(PriorityBasedMemory(64), (fov,fov,1), nov_thresh=0.25, novelty_loss_type='MSE', train_epochs_per_iter=1)
    print(Curiosity(map=map, brain=brain).get_from_position((fov, fov)))
    print(Random(map=map).get_from_position((fov, fov)))
    print(Linear(map=map).get_from_position((fov, fov)))

    agent = Agent(Curiosity(map=map, brain=brain), (2000,1000))
    agent.step()
    print("novelty")
    print(agent.get_path_novelty())
