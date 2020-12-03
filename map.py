from PIL import Image
from pathlib import Path
from IPython.display import Image as show
from typing import List, Tuple

class Map:
    def __init__(self, filepath: str, fov: int, sqrtGrains: int):
        self.fov = fov
        self.sqrtGrains = sqrtGrains

        self.img = Image.open(Path(filepath))
        # Convert the image to greyscale
        self.img = self.img.convert('L')

    def _in_map(self, position: Tuple[int]):
        valid_x = (position[0] < self.img.width and position[0] >= 0)
        valid_y = (position[1] < self.img.height and position[1] >= 0)

        return valid_x and valid_y

    # Finds coordinats from the given point in a way that assumes that the rover's position
    # is in the space between four pixels at the center of the grid, instead of offset one
    # to the left and one down
    # These are measured using an inverted y axis
    def _down(self, y_pos: int, distance: int):
        return y_pos - distance

    def _up(self, y_pos: int, distance: int):
        return y_pos + distance - 1

    def _left(self, y_pos: int, distance: int):
        return y_pos - distance + 1

    def _right(self, y_pos: int, distance: int):
        return y_pos + distance

    # Measured from the point of view of the inverted y
    # PIL refers to these in terms of the non-inverted y
    def full_view(self, position: Tuple[int]):
        lb_coordinates = (self._left(
            position[0], self.fov), self._down(position[1], self.fov))
        rt_coordinates = (self._right(
            position[0], self.fov+1), self._up(position[1], self.fov+1))
        # Need to add +1 because PIL doesn't take coordinates, it takes the length to travel
        # along each path

        return self.img.crop((*(lb_coordinates), *(rt_coordinates)))

    def get_fov(self, position: Tuple[int]):
        combined = self.full_view(position)

        # Visual top left
        vtl = combined.crop((0, 0, self.fov, self.fov))
        # Visual bottom left
        vbl = combined.crop((0, self.fov, self.fov, 2*self.fov))
        # Visual top right
        vtr = combined.crop((self.fov, 0, 2*self.fov, self.fov))
        # Visual bottom right
        vbr = combined.crop((self.fov, self.fov, 2*self.fov, 2*self.fov))

        return [
            [vtl, vtr],
            [vbl, vbr]
        ]

    def clean_directions(self, c: List[List[Tuple[int]]]):
        # Simpler to just do it this way since we only have four positions
        # Checks to see if the FOV would be off the screen if it took a step in that direction
        
        output = [ [None, None], [None, None] ]

        # Visual top left, which is bottom left in PIL
        output[0][0] = self._in_map(
            (self._left(c[0][0][0], self.fov), self._down(c[0][0][1], self.fov)))

        # Visual top right, which is bottom right in PIL
        output[0][1] = self._in_map(
            (self._right(c[0][1][0], self.fov), self._down(c[0][1][1], self.fov)))

        # Visual bottom left, which is top left in PIL
        output[1][0] = self._in_map(
            (self._left(c[1][0][0], self.fov), self._up(c[1][0][1], self.fov)))

        # Visual bottom right, which is top right in PIL
        output[1][1] = self._in_map(
            (self._right(c[1][1][0], self.fov), self._up(c[1][1][1], self.fov)))

        return output
