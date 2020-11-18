from PIL import Image
from pathlib import Path
import map_helpers as mh

class Map:
	"""
	Map class that creates instances of the terrain map that
	the model will work on

	__init__ : 
		initialize an instance of the given map and store fov and sqrtGrains

	get_fov(position): 
		returns a list of grains (sub-images) with radius fov given the position of the model on the map

	clean_directions(coordinates): 
		return a boolean list that corresponds to whether the model can move to the coordinates specified by the argument

	"""
	
	def __init__(self, filepath: str, fov: int, sqrtGrains: int):
		"""
		Parameters:

			filepath: the input terrain map -- can be a jpg or png
			fov: radius of the field-of-view
			sqrtGrains: The square root of the number of grains (sub-squares) in the fov


		Returns:

			Initializes an object with:

				The image from filepath (in greyscale),
				fov,
				sqrtGrains

		"""

		self.fov = fov
		self.sqrtGrains = sqrtGrains

		
		img = Image.open(Path(filepath))
		grey_scale = mh.is_grey_scale(img) # check if the image is greyscale

		if not grey_scale:
			img = img.convert('L') # convert image to greyscale

		self.img = img



	def get_fov(self, position: tuple):
		"""
		Parameter:

			position: Position of the rover on the map -- a tuple expected in (column, row)

		Returns:

			A list of the grains that have been split from the img with radius fov, i.e.
			the legal squares that the rover can go to in fov steps

		"""


		# Making sure the position is valid, i.e. at least fov pixels away from the edges
		is_valid = mh.is_valid_position(self.img, self.fov, position)

		if is_valid:

			width, height = self.img.size
		
			rover_position = mh.find_sitting_pixels(position, width, height)
			

			############ Crop the images to get grains of radius fov ##################

			## Get the (left, upper, right, lower) coordinates for each square the rover sits on

			all_coords = mh.find_coordinates(rover_position, self.fov, width, height)

			
			## Crop and return grains ##

			grains = []
			for coord in all_coords:
				if coord != None:

					if coord[0] != coord[2] and coord[1] != coord[3]:
						# For ex. (0, 534, 300, 534) --> the 534th row of the image b/w columns 0 and 300
						# Crop doesn't work for row/column "vectors"
						# It should never go in here based on how clean_directions is implemented, but just in case
						grains.append(self.img.crop(coord))


			for i in grains:
				i.show()

			
			return grains


		else:

			raise ValueError("Invalid position. Should be atleast [fov] pixels away from image edge.")




	def clean_directions(self, coordinates: list):
		"""
		Parameters: A list of tuples that represent coordinates

		Returns: A boolean array corresponding to each coordinate that indicates whether the model can move to that coordinate
		"""

		width, height = self.img.size
		valid_directions = []

		for pixel in coordinates:
			is_valid = mh.is_valid_position(self.img, self.fov, pixel)

			if is_valid:
				valid_directions.append(True)
			else:
				# Coordinate is closer than [fov] pixels to the edge #
				valid_directions.append(False)


		return valid_directions







x = Map("x.jpg", 150, 5)


# num_rows < num_cols in test image
# width = 800, height = 534
x.get_fov((600, 375))

x.clean_directions([(0, 0), (800, 534), (534, 800), (800, 535), (801, 534), (-1, 534), (200, -1), (200, 300)])




