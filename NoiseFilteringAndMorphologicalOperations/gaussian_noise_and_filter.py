import cv2
import numpy as np

gaussian_filter = [[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
					[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
					[0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
					[0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
					[0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]


def add_gaussian_noise(image, mean, sigma):
	rows, cols, channels = image.shape
	means = (mean, mean, mean)
	sigmas = (sigma, sigma, sigma)
	gauss = np.zeros((rows, cols, channels), np.uint8)
	cv2.randn(gauss, means, sigmas)
	noisy = image + gauss
	return noisy


def apply_gaussian_filter(image):
	rows, cols, channels = image.shape
	result = np.zeros((rows, cols, channels), np.uint8)
	for i in range(2, rows - 2):
		for j in range(2, cols - 2):
			r = g = b = 0
			for x in range(-2, 3):
				for y in range(-2, 3):
					b += (image.item(i + x, j + y, 0) * gaussian_filter[2 + x][2 + y])
					g += (image.item(i + x, j + y, 1) * gaussian_filter[2 + x][2 + y])
					r += (image.item(i + x, j + y, 2) * gaussian_filter[2 + x][2 + y])
			result.itemset((i, j, 0), b)
			result.itemset((i, j, 1), g)
			result.itemset((i, j, 2), r)
	return result


def main():
	cv2.namedWindow('Original IMG', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Original IMG', 0, 0)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	cv2.imshow('Original IMG', original_img)

	cv2.namedWindow('Gaussian Noisy Image Sigma = 15%', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Gaussian Noisy Image Sigma = 15%', 1000, 0)

	noisy = add_gaussian_noise(original_img, 0, 255*0.15)
	cv2.imshow('Gaussian Noisy Image Sigma = 15%', noisy)

	cv2.namedWindow('Gaussian Filter 5x5 Sigma = 1', cv2.WINDOW_NORMAL)
	cv2.moveWindow('Gaussian Filter 5x5 Sigma = 1', 1000, 1000)

	filtered = apply_gaussian_filter(noisy)
	cv2.imshow('Gaussian Filter 5x5 Sigma = 1', filtered)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
