import cv2
import numpy as np
import math

# RGB to HSV conversion and vice-versa


def get_bgr_of_pixel(img, x, y):
	r = img.item(x, y, 2)
	g = img.item(x, y, 1)
	b = img.item(x, y, 0)
	return r, g, b


def get_hsv_of_pixel(img, x, y):
	h = img.item(x, y, 0)
	s = img.item(x, y, 1)
	v = img.item(x, y, 2)
	return h, s, v


def convert_bgr_to_hsv(r, g, b):

	# Convert from 8-bit integers to floats
	byte_to_float = 1.0 / 255.0
	r, g, b = r * byte_to_float, g * byte_to_float, b * byte_to_float

	c_min = min(r, g, b)
	c_max = max(r, g, b)

	v = c_max
	delta = c_max - c_min

	# if both min and max are same (when R, G and B are same), delta goes to zero
	if delta == 0:
		h, s = 0, 0
	else:
		# Make sure its not pure black
		if c_max != 0:
			s = float(delta) / float(c_max)
			# Make the hues between 0.0 to 1.0 instead of 6.0
			angle_to_unit = float(1) / float(6.0 * delta)

			if r == c_max:
				h = float(g - b) * angle_to_unit  # Between Yellow and Magenta
			elif g == c_max:
				h = float(float(2) / float(6)) + float(b - r) * angle_to_unit  # Between Cyan and Yellow
			else:
				h = float(float(4) / float(6)) + float(r - g) * angle_to_unit  # Between Magenta and Cyan

			# Wrap outlier hues around the circle
			if h < 0.0:
				h += 1.0
			if h > 1.0:
				h -= 1.0
		else:
			h, s = 0, 0

	# Convert the floats to 8-bit integers
	h = int(0.5 + h * 255.0)
	s = int(0.5 + s * 255.0)
	v = int(0.5 + v * 255.0)

	# Clip the values to make sure it fits within 8 bits.
	if h > 255:
		h = 255
	if h < 0:
		h = 0
	if s > 255:
		s = 255
	if s < 0:
		s = 0
	if v > 255:
		v = 255
	if v < 0:
		v = 0

	return h, s, v


def convert_hsv_to_bgr(bh, bs, bv):
	byte_to_float = 1.0 / 255.0
	fh, fs, fv = bh * byte_to_float, bs * byte_to_float, bv * byte_to_float
	fr, fg, fb = 0.0, 0.0, 0.0
	# Achromatic - Grey
	if bs == 0:
		fr = fg = fb = fv
	else:
		if fh >= 1.0:
			fh = 0.0

		fh *= 6.0
		fi = math.floor(fh)
		ii = int(fh)
		ff = fh - fi

		p = fv * (1.0 - fs)
		q = fv * (1.0 - fs * ff)
		t = fv * (1.0 - fs * (1.0 - ff))

		if ii == 0:
			fr = fv
			fg = t
			fb = p
		elif ii == 1:
			fr = q
			fg = fv
			fb = p
		elif ii == 2:
			fr = p
			fg = fv
			fb = t
		elif ii == 3:
			fr = p
			fg = q
			fb = fv
		elif ii == 4:
			fr = t
			fg = p
			fb = fv
		elif ii == 5 or ii == 6:
			fr = fv
			fg = p
			fb = q

	br = int(fr * 255.0)
	bg = int(fg * 255.0)
	bb = int(fb * 255.0)

	if br > 255:
		br = 255
	if br < 0:
		br = 0
	if bg > 255:
		bg = 255
	if bg < 0:
		bg = 0
	if bb > 255:
		bb = 255
	if bb < 0:
		bb = 0

	return br, bg, bb


def convert_bgr_to_hsv_img(img):
	dimension = img.shape
	rows, cols, channels = dimension
	hsv_img = np.zeros(dimension, np.uint8)

	# Accessing the G, B and R channels of each pixel and converting it to HSV
	for i in range(rows):
		for j in range(cols):
			r, g, b = get_bgr_of_pixel(img, i, j)
			hue, sat, val = convert_bgr_to_hsv(r, g, b)
			hsv_img.itemset((i, j, 0), hue)
			hsv_img.itemset((i, j, 1), sat)
			hsv_img.itemset((i, j, 2), val)

	return hsv_img


def convert_hsv_to_bgr_img(img):
	dimension = img.shape
	rows, cols, channels = dimension
	bgr_img = np.zeros(dimension, np.uint8)

	for i in range(rows):
		for j in range(cols):
			h, s, v = get_hsv_of_pixel(img, i, j)
			r, g, b = convert_hsv_to_bgr(h, s, v)
			bgr_img.itemset((i, j, 0), b)
			bgr_img.itemset((i, j, 1), g)
			bgr_img.itemset((i, j, 2), r)

	return bgr_img


def main():
	# Creating windows
	cv2.namedWindow('Original IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('Original IMG', 0, 0)

	cv2.namedWindow('HSV IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('HSV IMG', 1000, 0)

	cv2.namedWindow("OpenCV Converted IMG", cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('OpenCV Converted IMG', 0, 1000)

	# Generating Images
	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)

	hsv_img = convert_bgr_to_hsv_img(original_img)
	print "{}, {}, {}".format(hsv_img.item(10,10,0), hsv_img.item(10,10,1),hsv_img.item(10,10,2))
	cv2.imshow('HSV IMG', hsv_img)

	new_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
	print "{}, {}, {}".format(new_img.item(10, 10, 0), new_img.item(10, 10, 1), new_img.item(10, 10, 2))
	cv2.imshow('OpenCV Converted IMG', new_img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
