"""
File that has helper functions for the Map class

1. is_grey_scale(img) : Checks whether the given image is greyscale or color

"""



def is_grey_scale(img):
	"""
	Parameters:
		img: the PIL image object

	Returns: A boolean indicating whether img is greyscale or not

	"""

	w, h = img.size
	for i in range(0, w):
		for j in range(0, h):
			r, g, b = img.getpixel((i, j))
			if r != g != b:
				return False


	return True



def find_sitting_pixels(position, width, height):
	"""
	Parameters: the position of the rover and width and height of the image

	Returns: A dictionary that represents the 4 pixels that the rover is "sitting" on
			 of the form {"position":(x, y), "above": (e, f), "right": (g, h), "top_right": (a, s)}

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
	Parameters: A dictionary of the pixels that the rover is "sitting" on, i.e. the one returned by find_sitting_pixels,
				The fov, width, height of image

	Returns: A list of the cropping coordinates for each of the (max) four grains

	"""

	## Get the (left, upper, right, lower) cropping coordinates for each square the rover sits on

	if rover_position["position"] != None:
		position = rover_position["position"]
		crop_position_coords = (max(0, position[0] - fov + 1), position[1], position[0], min(height, position[1] + fov - 1))
	else:
		crop_position_coords = None

	if rover_position["above"] != None:
		above = rover_position["above"]
		crop_above_coords = (max(0, above[0] - fov + 1), max(0, above[1] - fov + 1), above[0], above[1])
	else:
		crop_above_coords = None

	if rover_position["right"] != None:
		right = rover_position["right"]
		crop_right_coords = (right[0], right[1], min(width, right[0] + fov - 1), min(height, right[1] + fov - 1))
	else:
		crop_right_coords = None

	if rover_position["top_right"] != None:
		top_right = rover_position["top_right"]
		crop_top_right_coords = (top_right[0], max(0, top_right[1] - fov + 1), min(width, top_right[0] + fov - 1), top_right[1])
	else:
		crop_top_right_coords = None

	
	all_coords = [crop_position_coords, crop_above_coords, crop_right_coords, crop_top_right_coords]


	return all_coords