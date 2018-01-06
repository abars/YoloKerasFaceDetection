#Recognize gender from camera face

import os
import cv2
import sys
import numpy as np

import plaidml.keras
plaidml.keras.install_backend()

from keras.models import load_model
from keras.preprocessing import image

MODEL_HDF5='train_vgg16.hdf5'
#MODEL_HDF5='train_small_cnn.hdf5'

if(MODEL_HDF5 == 'train_vgg16.hdf5'):
	IMAGE_SIZE = 224
elif(MODEL_HDF5 == 'train_small_cnn.hdf5'):
	IMAGE_SIZE = 32
else:
	raise Exception('invalid model name')

model = load_model(MODEL_HDF5)
model.summary()

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
mirror = False

while True:
	ret, frame = cap.read() #BGR
	if mirror is True:
		frame = frame[:,::-1]

	target_image = frame

	gray_image = cv2.cvtColor(target_image,cv2.COLOR_BGR2GRAY)

	faces = classifier.detectMultiScale(gray_image)

	for i, (x,y,w,h) in enumerate(faces):
		margin=w/4

		x2=x-margin
		y2=y-margin
		w2=w+margin
		h2=h+margin

		if(x2<0):
			x2=0
		if(y2<0):
			y2=0
		if(w2>=target_image.shape[0]):
			w2=target_image.shape[0]-1
		if(h2>=target_image.shape[1]):
			h2=target_image.shape[1]-1

		face_image = target_image[y2:y2+h2, x2:x2+w2]

		img = cv2.resize(face_image, (IMAGE_SIZE,IMAGE_SIZE))
		img = img[::-1, :, ::-1].copy()	#BGR to RGB

		img = np.expand_dims(img, axis=0)
		img = img / 255.0

		pred = model.predict(img)[0]
		print(pred)

		prob = np.max(pred)
		cls = pred.argmax()

		lines=open('agegender_words.txt').readlines()
		print prob, cls, lines[cls]

		cv2.rectangle(target_image, (x2,y2), (x2+w2,y2+h2), color=(0,0,255), thickness=3)
		cv2.putText(target_image, str(prob)+lines[cls], (x2,y2+h2+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));

	cv2.imshow("agegender", target_image)

	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

