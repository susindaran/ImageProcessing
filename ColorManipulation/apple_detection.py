import cv2
import numpy as np
import color_space_conversion


# Red apple detection


class ImageFormat:
	def __init__(self):
		pass
	BGR = 1
	HSV = 2


def in_range(r, g, b, factor = 0):
	if (100 + factor < r < 256) and (0 <= g < 120 + (factor * 0.8)) and (0 <= b < 175 + (factor * 0.5)):
		return True
	return False


# Finding the biggest region of red pixels i.e., finding the apple
#
# Going to traverse row by row to find the row with most consecutive red pixels
# and then gonna traverse column by column to find the column with the most
# consecutive red pixels.
#
# Here "consecutive" is a bit forgiving - tolerance is used
#
# After finding both the axes, we can draw a rectangle around them.
def detect_red_apple(img, img_format, brightness_factor = 0):
	rows, cols, channels = img.shape

	# Row traversal
	row = 0
	final_h_start, final_h_end = -1, -1
	for y in range(rows):
		tolerance = 0
		# Longest section
		h_start, h_end = -1, -1
		# Placeholders for each section in the row
		ts, te = -1, -1
		started = False
		for x in range(cols):
			if img_format == ImageFormat.BGR:
				r, g, b = color_space_conversion.get_bgr_of_pixel(img, y, x)
				is_in_range = in_range(r, g, b, brightness_factor)
			else:
				h, s, v = color_space_conversion.get_hsv_of_pixel(img, y, x)
				is_in_range = h < 24 or h > 200

			if is_in_range and tolerance < 10:
				if not started:
					ts = x
					started = True
				te = x
			# Terminating the section
			elif tolerance >= 10:
				tolerance = 0
				started = False
				# Saving the progress if new length is achieved
				if (te - ts) > (h_end - h_start):
					h_start = ts
					h_end = te
			elif started:
				tolerance += 1

		# Saving the row and the line's start and stop if new length is achieved
		if (h_end - h_start) > (final_h_end - final_h_start):
			row = y
			final_h_end = h_end
			final_h_start = h_start

	# Column traversal
	col = 0
	final_v_start, final_v_end = -1, -1
	for x in range(cols):
		tolerance = 0
		# Longest section
		v_start, v_end = -1, -1
		# Placeholders for each section in the column
		ts, te = -1, -1
		started = False
		for y in range(rows):
			if img_format == ImageFormat.BGR:
				r, g, b = color_space_conversion.get_bgr_of_pixel(img, y, x)
				is_in_range = in_range(r, g, b, brightness_factor)
			else:
				h, s, v = color_space_conversion.get_hsv_of_pixel(img, y, x)
				is_in_range = h < 24 or h > 200

			if is_in_range and tolerance < 10:
				if not started:
					ts = y
					started = True
				te = y
			elif tolerance >= 10:
				tolerance = 0
				started = False
				if (te - ts) > (v_end - v_start):
					v_start = ts
					v_end = te
			elif started:
				tolerance += 1
		# Saving the column and the line's start and stop if new length is achieved
		if (v_end - v_start) > (final_v_end - final_v_start):
			col = x
			final_v_start = v_start
			final_v_end = v_end

	return (final_h_start, row), (final_h_end, row), (col, final_v_start), (col, final_v_end)


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('Original IMG', 0, 0)

	cv2.namedWindow('HSV IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('HSV IMG', 1000, 0)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	hsv_img = color_space_conversion.convert_bgr_to_hsv_img(original_img)

	#################################################
	# For Development - Remove in the final version #
	#################################################
	dimension = original_img.shape
	new_img = np.zeros(dimension, np.uint8)
	rows, cols, channels = dimension
	# bright_img = q1.increase_brightness(original_img, 50)
	# Creating new image with only red(shades of red) pixels.
	for y in range(rows):
		for x in range(cols):
			r, g, b = color_space_conversion.get_bgr_of_pixel(original_img, y, x)
			# h, s, v = q2.convert_bgr_to_hsv(r, g, b)

			# The darker red has a hue value below 25 and lighter red has a hue value
			# above 200. Pixels with hue value between 25 and 199 are made black.
			if not in_range(r, g, b):
				r, g, b = 0, 0, 0

			new_img.itemset((y, x, 0), b)
			new_img.itemset((y, x, 1), g)
			new_img.itemset((y, x, 2), r)

	cv2.namedWindow('Red Apple', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('Red Apple', 0, 1000)
	cv2.imshow('Red Apple', new_img)
	#################################################

	h_start, h_end, v_start, v_end = detect_red_apple(original_img, ImageFormat.BGR)
	print 'Original Image:\nh_start: {}, h_end: {}\nv_start: {}, v_end: {}'.format(h_start, h_end, v_start, v_end)
	cv2.rectangle(original_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	cv2.imshow('Original IMG', original_img)

	h_start, h_end, v_start, v_end = detect_red_apple(hsv_img, ImageFormat.HSV)
	print '\nHSV Image:\nh_start: {}, h_end: {}\nv_start: {}, v_end: {}'.format(h_start, h_end, v_start, v_end)
	cv2.rectangle(hsv_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	cv2.imshow('HSV IMG', hsv_img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
