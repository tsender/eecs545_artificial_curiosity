"""
File that has helper functions for the Map class

1. is_grey_scale(img) : Checks whether the given image is greyscale or color

"""



def is_grey_scale(img):
	"""
	Parameters:
		img: the PIL image object

	Returns a boolean indicating whether img is greyscale or not

	"""

	w, h = img.size
	for i in range(0, w):
		for j in range(0, h):
			r, g, b = img.getpixel((i, j))
			if r != g != b:
				return False


	return True