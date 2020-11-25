"""
File that has helper functions for the Map class

1. is_grey_scale(img) : Checks whether the given image is greyscale or color
2. is_valid_position(img, fov, position): Checks whether the given position is valid or not
3. find_sitting_pixels(position, width, height): Finds the pixel that the rover is "sitting" on
4. find_coordinates(rover_position, fov, width, height): Finds the cropping coordinates that create grains

"""



def is_grey_scale(img):
	"""
	Parameter:
		img: the PIL image object

	Returns: 
		A boolean indicating whether img is greyscale or not

	"""

	w, h = img.size
	for i in range(0, w):
		for j in range(0, h):
			r, g, b = img.getpixel((i, j))
			if r != g != b:
				return False


	return True



def is_valid_position(img, fov, position):
	"""
	Parameters: 
		The image, fov and position passed to get_fov in map.py

	Returns: 
		A boolean indicating whether it is a valid position or not, i.e. atleast [fov] pixels away from image edge

	"""

	width, height = img.size

	column, row = position

	if column >= fov - 1 and column < width - fov and row > fov - 1 and row <= height - fov:

		return True

	else:

		return False



def find_sitting_pixels(position, width, height):
	"""
	Parameters: 
		position: the position of the rover
		width: width (number of columns) of the image
		height: height (number of rows) of the image

	Returns: 
		A dictionary that represents the 4 pixels that the rover is "sitting" on

	"""

	column, row = position


	#### We use top-right as the offset to find the other 3 squares the rovers "sits" on  #####

		## Check if the squares the rover sits on are within the boundary of the map ##

	if row - 1 >= 0:

		above = (column, row-1)

	else:

		above = None

	if column + 1 <= width:

		right = (column+1, row)

	else:

		right = None

	

	if column == width or row == 0:

		# rover is in the top row or the right most column of the map

		top_right = None

	else:

		top_right = (column + 1, row - 1)


	
	# rover sits on position, above, right, top_right
	rover_position = {"position":position, "above":above, "right":right, "top_right":top_right}


	return rover_position



def find_coordinates(rover_position, fov, width, height):
	"""
	Parameters: 
		rover_position: A dictionary of the pixels that the rover is "sitting" on, i.e. the one returned by find_sitting_pixels
		fov: the fov
		width: the width of the image
		height: the height of the image

	Returns: 
		A list of the cropping coordinates for each of the (max) four grains

	"""

	## Get the (left, upper, right, lower) cropping coordinates for each square the rover sits on

	if rover_position["position"] != None:
		position = rover_position["position"]
		crop_position_coords = (max(0, position[0] - fov + 1), position[1], position[0] + 1, min(height, position[1] + fov))
	else:
		crop_position_coords = None

	if rover_position["above"] != None:
		above = rover_position["above"]
		crop_above_coords = (max(0, above[0] - fov + 1), max(0, above[1] - fov + 1), above[0] + 1, above[1] + 1)
	else:
		crop_above_coords = None

	if rover_position["right"] != None:
		right = rover_position["right"]
		crop_right_coords = (right[0], right[1], min(width, right[0] + fov), min(height, right[1] + fov))
	else:
		crop_right_coords = None

	if rover_position["top_right"] != None:
		top_right = rover_position["top_right"]
		crop_top_right_coords = (top_right[0], max(0, top_right[1] - fov + 1), min(width, top_right[0] + fov), top_right[1] + 1)
	else:
		crop_top_right_coords = None

	
	all_coords = [crop_position_coords, crop_above_coords, crop_right_coords, crop_top_right_coords]


	return all_coords
