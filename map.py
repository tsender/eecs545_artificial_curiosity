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

		print(grey_scale)

		if not grey_scale:
			img = img.convert('L') # convert image to greyscale


		#print(img.getpixel((0, 0)))
		self.img = img



	def get_fov(self, position: tuple):
		"""
		Parameter:

			position: Position of the rover on the map -- a tuple

		Returns:

			A list of the grains that have been split from the img with radius fov, i.e.
			the legal squares that the rover can go to in fov steps

		"""





	def clean_directions(self, coordinates: tuple):
		pass







#x = Map("x.jpeg", 2, 5)






