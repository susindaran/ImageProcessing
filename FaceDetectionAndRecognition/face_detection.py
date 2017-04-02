import cv2
import numpy as np
import os
from PIL import Image


def get_images_and_labels(path, face_cascade):
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	images = []
	labels = []

	for image_path in image_paths:
		gray = Image.open(image_path).convert('L')
		image = np.array(gray, 'uint8')
		label = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
		faces = face_cascade.detectMultiScale(image)

		for (x, y, w, h) in faces:
			images.append(image[y: y + h, x: x + w])
			labels.append(label)

	return images, labels


def get_faces(video_capture, face_cascade):
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = np.array(gray, 'uint8')

	faces = face_cascade.detectMultiScale(image)
	return frame, image, faces


def main():
	cascade_path = "haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(cascade_path)
	recognizer = cv2.face.createLBPHFaceRecognizer()

	video_capture = cv2.VideoCapture(0)

	print "Loading faces from dataset"
	training_images, training_images_labels = get_images_and_labels('faces', face_cascade)

	training_image_capture_count = 15
	print "Recognizing first person"
	while True:
		frame, image, faces = get_faces(video_capture, face_cascade)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			if training_image_capture_count > 0:
				training_images.append(image[y: y + h, x: x + w])
				training_images_labels.append(25)
				training_image_capture_count -= 1

		cv2.putText(frame, 'Training 1st person', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	training_image_capture_count = 15
	print "Recognizing second person"
	while True:
		frame, image, faces = get_faces(video_capture, face_cascade)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			if training_image_capture_count > 0:
				training_images.append(image[y: y + h, x: x + w])
				training_images_labels.append(26)
				training_image_capture_count -= 1

		cv2.putText(frame, 'Training 2nd person', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	recognizer.train(training_images, np.array(training_images_labels))

	while True:
		frame, image, faces = get_faces(video_capture, face_cascade)

		for (x, y, w, h) in faces:
			predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
			if predicted == 25:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)

		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
