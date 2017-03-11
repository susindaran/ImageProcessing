import cv2
import color_space_conversion, adjust_brightness_rgb


def mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		r, g, b = color_space_conversion.get_bgr_of_pixel(param, y, x)
		h, s, v = color_space_conversion.convert_bgr_to_hsv(r, g, b)
		print 'r: {}, g: {}, b: {}   |   h : {}, s: {}, v: {}'.format(r, g, b, h, s, v)


def hsv_mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		h, s, v = param.item(y, x, 0), param.item(y, x, 1), param.item(y, x, 2)
		print 'h : {}, s: {}, v: {}'.format(h, s, v)


def main():
	img = cv2.imread('../Data/color_92331.png', cv2.IMREAD_UNCHANGED)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("img", hsv_mouse_callback, param = hsv)
	cv2.imshow('img', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
