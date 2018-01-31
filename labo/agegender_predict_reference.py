# ----------------------------------------------
# Reference age and gender classifier
# ----------------------------------------------

import caffe
import cv2
import sys
import numpy as np

import plaidml.keras
plaidml.keras.install_backend()

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model

import keras2caffe

# ----------------------------------------------
# converting
# ----------------------------------------------

IMAGE_SIZE = 227

net_age  = caffe.Net('deploy_age.prototxt', 'age_net.caffemodel', caffe.TEST)
net_gender  = caffe.Net('deploy_gender.prototxt', 'gender_net.caffemodel', caffe.TEST)

# ----------------------------------------------
# data
# ----------------------------------------------

img = cv2.imread('agegender/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg')
#img = cv2.imread('agegender/annotations/agegender/validation/11_15-20_f/landmark_aligned_face.290.11593366185_289173a738_o.jpg')

img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data -= (104,117,123)

# ----------------------------------------------
# verify
# ----------------------------------------------

data = data.transpose((0, 3, 1, 2))

out = net_age.forward_all(data = data)
pred = out['prob']
prob = np.max(pred)
cls = pred.argmax()
lines=open('words/agegender_age_words.txt').readlines()
print prob, cls, lines[cls]

out = net_gender.forward_all(data = data)
pred = out['prob']
prob = np.max(pred)
cls = 1-pred.argmax()
lines=open('words/agegender_gender_words.txt').readlines()
print prob, cls, lines[cls]
