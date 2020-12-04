from PIL import Image
from pathlib import Path
# from IPython.display import Image as show
from typing import List, Tuple

class Map:
    """
    This class will hold some basic information about the environment our agent is training in,
    which allows us to train multiple agents in the same environment. It's main purpose is to
    act as a wrapper for the image we will be using as our domain

    It is important to note that the map uses an inverted y axis due to the graphics backend
    """

    def __init__(self, filepath: str, fov: int, sqrtGrains: int, greyscale: bool = True):
        """
        Params
        ------
        filepath: str
            The path to the image that we want to use as our domain. It will be loaded automatically

        fov: int
            The radius of our agent's field of view (FOV). This is how far they can see, as well as
            how far they'll stay away from the walls (because we don't want them to be able to go
            off the map).

        sqrtGrains: int
            The square root of the number of grains, or sub-images, that we want the agent to use.
            This number can also be thought of as the number of grains that are along one side of
            the square that makes ip the FOV (hence thew square root).

            The number of grains is effectively the number of sub-patches that we create for the
            model to use for navigation. Right now, we have informally agreed to limit them
            to two for simplicity. This will not be the case in the future

        greyscale: bool
            Whether the image should be converted to greyscale
        """

        # Assign the variables
        self.fov = fov
        self.sqrtGrains = sqrtGrains

        # Load the image into memory
        self.img = Image.open(Path(filepath))

        # Convert the image to greyscale
        if(greyscale):
            self.img = self.img.convert('L')


    def _in_map(self, position: Tuple[int]):
        """
        Determines whether the given position is in the map or not.

        Params
        ------

        position: Tuple[int]
            The position that we want to check against the map
        """

        # Makes sure that it's less than height/width and more than 0
        valid_x = (position[0] < self.img.width and position[0] >= 0)
        valid_y = (position[1] < self.img.height and position[1] >= 0)

        return valid_x and valid_y

    

    """
    This explaination applies to the next four functions equally, so I'm going to put it
    here:

    Our rover sits on a specific pixel. However, we want to be able to split its field of 
    view perfectly in half. As a result, we slightly offset our field of view so that the
    rover is sitting in the center pixel of one of the center pixels. However, this means
    that calculating positions based on the rover's position is a little trick, since
    we have to act like the rover is actually between the four center pixels (think of
    the rover as being in the bottom left square of the text "[+]", but us acting as if
    it's on the center of the plus sign to make sure that its behaviour is balanced).

    These functions are for transforming direction information from the saved position's
    point of view to behave as if it's coming from the center of 4 pixels. This means that
    there are slightly different offsets for each function, but we can treat them the same
    and they will resolve the difference.

    Note that we are using an inverted Y axis.
    """

    def _down(self, y_pos: int, distance: int):
        """
        Find the pixel a given distance from the virtual agent location

        Params
        ------

        y_pos: int
            An index along the y axis that will be used as the starting point

        distance: int
            The distance we want to trayel along the y axis from that point.

        Returns
        -------

        An integer for the new location
        """
        return y_pos - distance


    def _up(self, y_pos: int, distance: int):
        """
        Find the pixel a given distance from the virtual agent location

        Params
        ------

        y_pos: int
            An index along the y axis that will be used as the starting point

        distance: int
            The distance we want to trayel along the y axis from that point.

        Returns
        -------

        An integer for the new location
        """
        return y_pos + distance - 1


    def _left(self, x_pos: int, distance: int):
        """
        Find the pixel a given distance from the virtual agent location

        Params
        ------

        x_pos: int
            An index along the x axis that will be used as the starting point

        distance: int
            The distance we want to trayel along the x axis from that point.

        Returns
        -------

        An integer for the new location
        """
        return x_pos - distance + 1


    def _right(self, x_pos: int, distance: int):
        """
        Find the pixel a given distance from the virtual agent location

        Params
        ------

        x_pos: int
            An index along the x axis that will be used as the starting point

        distance: int
            The distance we want to trayel along the x axis from that point.

        Returns
        -------

        An integer for the new location
        """
        return x_pos + distance


    def full_view(self, position: Tuple[int]):
        """
            This fuction will get the image that covers the entirety of
            the agent's FOV, This is so that we can see everything 
            that the agent has seen, as well as use this as an intermediate
            function later when we're getting individual grains
        """

        # Need to add +1 because PIL doesn't take coordinates, it takes the length to travel
        # along each path
        lb_coordinates = (self._left(
            position[0], self.fov), self._down(position[1], self.fov))

        rt_coordinates = (self._right(
            position[0], self.fov+1), self._up(position[1], self.fov+1))

        return self.img.crop((*(lb_coordinates), *(rt_coordinates)))


    def get_fov(self, position: Tuple[int]):
        """
        Gets the grains for the rover at the given position and returns them

        Params
        ------

        position: Tuple[int]
            The position that the rover is sitting at. This will be adgusted so that the
            rover is virtually in between four points.

        Returns
        -------

        List[List[Image.Image]]
            Returns the agent's field of view as a 2d array of images in the format
            <visual_top_left>,    <visual_top_right>
            <visual_bottom_left>, <visual_buttom_right>

            Visual refers to the fact that <visual_top_left> looks like it's in the
            top left, but according to PIL it's actually in the bottom left
        """
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
        """
        This function makes sure that the agent wouldn't be able to see out
        of the map if it moved to these positions. This is to prevent the
        agent from running off the map.

        For right now we assume that there are 4 grains being passed. Because
        we're only checking 4 positions, it's easier and more efficient to
        just unroll the lop and write them all out individually

        Params
        ------
        c: List[List[int]]
            A 2d list of positions that will be checked against the map.
            
        Returns
        -------
        List[List[bool]]
            A 2d list of booleans that tell whether a given direction is valid
        """
                
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
