import cv2
import numpy as np
import random


def add_salt_and_pepper_noise(image, density):
	rows, cols, channels = image.shape
	result = image.copy()
	threshold = 1 - density

	for i in range(rows):
		for j in range(cols):
			rnd = random.random()
			if rnd < density:
				result[i][j] = 0
			elif rnd > threshold:
				result[i][j] = 255

	return result


def apply_median_filter(image):
	rows, cols, channels = image.shape
	result = np.zeros((rows, cols, channels), np.uint8)

	members = [[0]*3]*9
	for i in range(1, rows-1):
		for j in range(1, cols-1):
			for x in range(-1, 2):
				for y in range(-1, 2):
					members[((x + 1) * 3) + (y + 1)] = [image.item(i + x, j + y, 0), image.item(i + x, j + y, 1), image.item(i + x, j + y, 2)]

			result.itemset((i, j, 0), np.median([x[0] for x in members]))
			result.itemset((i, j, 1), np.median([x[1] for x in members]))
			result.itemset((i, j, 2), np.median([x[2] for x in members]))

	return result


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Original IMG', 0, 0)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)

	cv2.namedWindow('Salt and Pepper Noise 0.02', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Salt and Pepper Noise 0.02', 0, 1000)

	noisy = add_salt_and_pepper_noise(original_img, 0.02)
	cv2.imshow('Salt and Pepper Noise 0.02', noisy)

	cv2.namedWindow('Median Filter', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Median Filter', 1000, 0)

	filtered = apply_median_filter(noisy)
	cv2.imshow('Median Filter', filtered)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
