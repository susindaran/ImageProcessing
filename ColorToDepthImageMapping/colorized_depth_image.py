import cv2
import numpy as np

inv_intrinsic_mat = [[0.0027289, 0, -0],
					[-7.1662e-06, 0.0027186, -0],
					[-0.71173, -0.56546, 1]]

transform_d_to_c = [[0.99998, 0.0262464, 0.0013162, 0],
					[0.0062361, 0.99997, 0.0046386, 0],
					[-0.0013491, -0.0046356, 0.99999, 0],
					[50.775, 11.994, -80.412, 1]]

intrinsic_mat = [[1027, 0, 0],
				[3.4052, 1029.9, 0],
				[671, 836.54, 1]]


def generate_colorized_image(depth_image, color_image):
	rows, cols = depth_image.shape

	colorized_image = np.zeros((rows, cols, 4), np.uint8)

	for u in range(rows):
		for v in range(cols):
			z = depth_image.item(u, v)
			if not z == 0:
				x = (float(u) * inv_intrinsic_mat[0][0] + float(v) * inv_intrinsic_mat[1][0] + float(1) * inv_intrinsic_mat[2][0]) * float(z)
				y = (float(u) * inv_intrinsic_mat[0][1] + float(v) * inv_intrinsic_mat[1][1] + float(1) * inv_intrinsic_mat[2][1]) * float(z)

				d_mat = [x, y, z, 1]
				c_mat = np.dot(d_mat, transform_d_to_c)

				color_mat = np.dot([c_mat[0] / c_mat[2], c_mat[1] / c_mat[2], 1], intrinsic_mat)
				cx = int(color_mat[0])
				cy = int(color_mat[1])

				if (0 < cx < 1080) and (0 < cy < 1920):
					colorized_image.itemset((u, v, 0), color_image.item(cx, cy, 0))
					colorized_image.itemset((u, v, 1), color_image.item(cx, cy, 1))
					colorized_image.itemset((u, v, 2), color_image.item(cx, cy, 2))

	return colorized_image


def get_3d_coords(x, y, z):
	arr = [0, 0, 0]
	if not z == 0:
		x = (float(x) * inv_intrinsic_mat[0][0] + float(y) * inv_intrinsic_mat[1][0] + float(1) * inv_intrinsic_mat[2][0]) * float(z)
		y = (float(x) * inv_intrinsic_mat[0][1] + float(y) * inv_intrinsic_mat[1][1] + float(1) * inv_intrinsic_mat[2][1]) * float(z)

		d_mat = [x, y, z, 1]
		c_mat = np.dot(d_mat, transform_d_to_c)
		arr = c_mat[0:3]

	return arr


def apply_mask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	rows, cols, dims = hsv.shape
	for y in range(rows):
		for x in range(cols):
			h = hsv.item(y, x, 0)
			if not(0 < h < 20 or 120 <= h <= 172):
				img.itemset((y, x, 0), 0)
				img.itemset((y, x, 1), 0)
				img.itemset((y, x, 2), 0)

	return img


def detect_balls(colorized_img):
	masked_img = apply_mask(colorized_img.copy())
	gray_1 = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

	circles = cv2.HoughCircles(gray_1, cv2.HOUGH_GRADIENT, 2.5, 20)
	print circles
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for (x, y, r) in circles:
			cv2.rectangle(colorized_img, (x - r, y - r), (x + r, y + r), (0, 128, 255), 2)

	return circles[0], circles[1]


def calculate_distance(first_position, second_position, depth_image_1, depth_image_2):
	a = get_3d_coords(first_position[0], first_position[1], depth_image_1.item(first_position[1], first_position[0]))
	b = get_3d_coords(second_position[0], second_position[1], depth_image_2.item(second_position[1], second_position[0]))

	return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]) + np.square(a[2] - b[2]))


def main():
	depth_image_1 = cv2.imread("depth_92331d.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
	color_image_1 = cv2.imread("color_92331.png", cv2.IMREAD_UNCHANGED)

	depth_image_2 = cv2.imread("depth_94764d.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
	color_image_2 = cv2.imread("color_94764.png", cv2.IMREAD_UNCHANGED)

	colorized_image_1 = generate_colorized_image(depth_image_1, color_image_1)
	first_big, first_small = detect_balls(colorized_image_1)
	cv2.namedWindow('First Colorized IMG', cv2.WINDOW_NORMAL)
	cv2.imshow('First Colorized IMG', colorized_image_1)

	colorized_image_2 = generate_colorized_image(depth_image_2, color_image_2)
	second_big, second_small = detect_balls(colorized_image_2)
	cv2.namedWindow('Second Colorized IMG', cv2.WINDOW_NORMAL)
	cv2.imshow('Second Colorized IMG', colorized_image_2)

	dist_big_ball = calculate_distance(first_big, second_big, depth_image_1, depth_image_2)
	dist_small_ball = calculate_distance(first_small, second_small, depth_image_1, depth_image_2)

	print "Distance travelled by big ball: {} mm".format(dist_big_ball)
	print "Distance travelled by small ball: {} mm".format(dist_small_ball)

	time_in_seconds = float(2433) / 1000

	big_ball_speed = dist_big_ball / time_in_seconds
	small_ball_speed = dist_small_ball / time_in_seconds

	print "Speed of big ball: {} mm/sec".format(big_ball_speed)
	print "Speed of small ball: {} mm/sec".format(small_ball_speed)
	print "Relative speed: {} mm/sec".format(big_ball_speed + small_ball_speed)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
