import cv2
import adjust_brightness_rgb, color_space_conversion, apple_detection, color_inversion


def display_window(window_name, image, x_coord, y_coord):
	cv2.imshow(window_name, image)
	cv2.moveWindow(window_name, x_coord, y_coord)


def main():
	#############################
	# Displaying Original Image #
	#############################
	cv2.namedWindow('Original IMG', cv2.WINDOW_AUTOSIZE)

	original_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
	display_window('Original IMG', original_img, 0, 0)

	cv2.waitKey(0)

	###############################
	# Displaying brightened Image #
	###############################
	cv2.namedWindow('Brightened IMG', cv2.WINDOW_AUTOSIZE)

	brightened_img = adjust_brightness_rgb.increase_brightness(original_img, 50)
	display_window('Brightened IMG', brightened_img, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyWindow('Brightened IMG')

	################################################
	# Displaying RGB to HSV color space conversion #
	################################################
	cv2.namedWindow('HSV IMG', cv2.WINDOW_AUTOSIZE)

	hsv_img = color_space_conversion.convert_bgr_to_hsv_img(original_img)
	display_window('HSV IMG', hsv_img, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.namedWindow('Brightened HSV IMG', cv2.WINDOW_AUTOSIZE)
	brightened_hsv_img = color_space_conversion.convert_bgr_to_hsv_img(brightened_img)
	display_window('Brightened IMG', brightened_img, 0, 0)
	display_window('Brightened HSV IMG', brightened_hsv_img, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()
	####################################
	# Detecting Red Apple in the Image #
	####################################
	apple_marked_img = original_img.copy()
	h_start, h_end, v_start, v_end = apple_detection.detect_red_apple(original_img, apple_detection.ImageFormat.BGR)
	cv2.rectangle(apple_marked_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	display_window('Original IMG', apple_marked_img, 0, 0)

	apple_marked_hsv_img = hsv_img.copy()
	h_start, h_end, v_start, v_end = apple_detection.detect_red_apple(hsv_img, apple_detection.ImageFormat.HSV)
	cv2.rectangle(apple_marked_hsv_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	display_window('HSV IMG', apple_marked_hsv_img, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()

	apple_marked_brightened_img = brightened_img.copy()
	h_start, h_end, v_start, v_end = apple_detection.detect_red_apple(brightened_img, apple_detection.ImageFormat.BGR, 50)
	cv2.rectangle(apple_marked_brightened_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	display_window('Brightened IMG', apple_marked_brightened_img, 0, 0)

	apple_marked_brightened_hsv_img = brightened_hsv_img.copy()
	h_start, h_end, v_start, v_end = apple_detection.detect_red_apple(brightened_hsv_img, apple_detection.ImageFormat.HSV)
	cv2.rectangle(apple_marked_brightened_hsv_img, (h_start[0], v_start[1]), (h_end[0], v_end[1]), (255, 0, 0), 3)
	display_window('Brightened HSV IMG', apple_marked_brightened_hsv_img, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()

	#################################
	# Converting Red apple to Green #
	#################################
	cv2.namedWindow('Red to Green - Original IMG', cv2.WINDOW_AUTOSIZE)

	red_to_green_apple = color_inversion.convert_red_to_green(original_img)
	display_window('Original IMG', original_img, 0, 0)
	display_window('Red to Green - Original IMG', red_to_green_apple, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.namedWindow('Red to Green - Brightened IMG', cv2.WINDOW_AUTOSIZE)

	red_to_green_brightened_apple = color_inversion.convert_red_to_green(brightened_img)
	display_window('Brightened IMG', brightened_img, 0, 0)
	display_window('Red to Green - Brightened IMG', red_to_green_brightened_apple, 1000, 0)

	key = cv2.waitKey(0)
	while key != ord('n'):
		key = cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
