import cv2
import numpy as np

# 1. Create a brightness adjusted image B1 from the image I by adding a constant factor 50.


def increase_brightness(img, factor):
	dimension = img.shape
	rows, cols, channels = dimension
	brightened_img = np.zeros(dimension, np.uint8)

	# Accessing the G, B and R channels of each pixel and adding the factor 50
	# while saturating the result to 255
	for i in range(rows):
		for j in range(cols):
			brightened_img.itemset((i, j, 0), min(img.item(i, j, 0) + factor, 255))
			brightened_img.itemset((i, j, 1), min(img.item(i, j, 1) + factor, 255))
			brightened_img.itemset((i, j, 2), min(img.item(i, j, 2) + factor, 255))

	return brightened_img


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Original IMG', 0, 0)

	cv2.namedWindow('Brightened IMG', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Brightened IMG', 1000, 0)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)

	brightened_img = increase_brightness(original_img, 50)
	cv2.imshow('Brightened IMG', brightened_img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
