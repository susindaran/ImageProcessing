import cv2
import numpy as np

import gaussian_noise_and_filter, median_filter, adjust_brightness


def count_connected_components(image, kernel_size, opened = True, open_iter = 5, dilate = True, dilate_iter = 1):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Threshold', 0, 1000)
	cv2.imshow('Threshold', thresh)

	if opened:
		second = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = open_iter)
	else:
		second = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = open_iter)

	cv2.namedWindow('Opening', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Opening', 1000, 0)
	cv2.imshow('Opening', second)

	if dilate:
		third = cv2.dilate(second, kernel, iterations = dilate_iter)
	else:
		third = cv2.erode(second, kernel, iterations = dilate_iter)

	output = cv2.connectedComponentsWithStats(third, 8, cv2.CV_32S)
	num_components = output[0]

	third = cv2.cvtColor(third, cv2.COLOR_GRAY2BGR)

	cv2.putText(third, 'Components: {}'.format(num_components), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	cv2.namedWindow('Dilating', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Dilating', 1000, 1000)
	cv2.imshow('Dilating', third)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def display_window(window_name, image, x_coord, y_coord):
	cv2.imshow(window_name, image)
	cv2.moveWindow(window_name, x_coord, y_coord)


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Original IMG', 0, 0)
	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)
	count_connected_components(original_img, 3)

	# Gaussian Noise
	cv2.namedWindow('Gaussian Noise', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Gaussian Noise', 0, 0)
	gaussian_noise = gaussian_noise_and_filter.add_gaussian_noise(original_img, 0, 255 * 0.15)
	cv2.imshow('Gaussian Noise', gaussian_noise)
	count_connected_components(gaussian_noise, 3, False, 3, False)

	# Gaussian Noise - Filtered
	cv2.namedWindow('Gaussian Noise - Filtered', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Gaussian Noise - Filtered', 0, 0)
	gaussian_noise_filtered = gaussian_noise_and_filter.apply_gaussian_filter(gaussian_noise)
	cv2.imshow('Gaussian Noise - Filtered', gaussian_noise_filtered)
	count_connected_components(gaussian_noise_filtered, 3)

	# Salt and Pepper
	cv2.namedWindow('Salt and Pepper', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Salt and Pepper', 0, 0)
	salt_and_pepper = median_filter.add_salt_and_pepper_noise(original_img, 0.02)
	cv2.imshow('Salt and Pepper', salt_and_pepper)
	count_connected_components(salt_and_pepper, 3, False, 2, False, 3)

	# Salt and Pepper - Filtered
	cv2.namedWindow('Salt and Pepper - Filtered', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Salt and Pepper - Filtered', 0, 0)
	salt_and_pepper_filtered = median_filter.apply_median_filter(salt_and_pepper)
	cv2.imshow('Salt and Pepper - Filtered', salt_and_pepper_filtered)
	count_connected_components(salt_and_pepper_filtered, 3)

	# Brightened
	cv2.namedWindow('Brightened', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Brightened', 0, 0)
	brightened = adjust_brightness.increase_brightness(original_img, 50)
	cv2.imshow('Brightened', brightened)
	count_connected_components(brightened, 3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
