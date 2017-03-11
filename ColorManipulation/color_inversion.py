import cv2
import numpy as np
import color_space_conversion

# Invert color of Apples


def convert_red_to_green(img):
	dimension = img.shape
	rows, cols, channels = dimension
	new_img = np.zeros(dimension, np.uint8)

	for y in range(rows):
		for x in range(cols):
			r, g, b = color_space_conversion.get_bgr_of_pixel(img, y, x)
			new_img.itemset((y, x, 0), b)
			new_img.itemset((y, x, 1), r)
			new_img.itemset((y, x, 2), g)

	return new_img


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('Original IMG', 0, 0)

	cv2.namedWindow('Swapped IMG', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('Swapped IMG', 1000, 0)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)

	swapped_img = convert_red_to_green(original_img)
	cv2.imshow('Swapped IMG', swapped_img)

	cv2. waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
