from PIL import Image
import os

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

		"""

		self.fov = fov
		self.sqrtGrains = sqrtGrains



	def get_fov(self, position: tuple):
		pass



	def clean_directions(self, coordinates: tuple):
		pass