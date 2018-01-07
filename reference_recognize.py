# ----------------------------------------------
# Reference gender from camera face
# ----------------------------------------------

import os
import cv2
import sys
import numpy as np

import plaidml.keras
plaidml.keras.install_backend()

from keras.models import load_model
from keras.preprocessing import image

import caffe

# ----------------------------------------------
# Models
# ----------------------------------------------

IMAGE_SIZE = 227

# ----------------------------------------------
# Classifier
# ----------------------------------------------

#classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

net_age  = caffe.Net('deploy_age.prototxt', 'age_net.caffemodel', caffe.TEST)
net_gender  = caffe.Net('deploy_gender.prototxt', 'gender_net.caffemodel', caffe.TEST)

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
		margin=0#w/4

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

		img = np.expand_dims(img, axis=0)
		img = img - 128

		img = img.transpose((0, 3, 1, 2))

		out = net_age.forward_all(data = img)
		pred_age = out['prob']
		prob_age = np.max(pred_age)
		cls_age = pred_age.argmax()
		lines_age=open('agegender_age_words.txt').readlines()

		out = net_gender.forward_all(data = img)
		pred_gender = out['prob']
		prob_gender = np.max(pred_gender)
		cls_gender = 1-pred_gender.argmax()
		lines_gender=open('agegender_gender_words.txt').readlines()

		cv2.rectangle(target_image, (x2,y2), (x2+w2,y2+h2), color=(0,0,255), thickness=3)
		cv2.putText(target_image, str(prob_age)+lines_age[cls_age], (x2,y2+h2+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
		cv2.putText(target_image, str(prob_gender)+lines_gender[cls_gender], (x2,y2+h2+32), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));

	cv2.imshow("agegender", target_image)
	#cv2.imwrite("agegender_result.jpg", target_image)

	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

