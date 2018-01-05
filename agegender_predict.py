# ----------------------------------------------
# Predict age gender classifier
# ----------------------------------------------

import caffe
import cv2
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

#MODEL_HDF5='train_small_cnn2.hdf5'
#MODEL_HDF5='train_small_cnn.hdf5'
MODEL_HDF5='train_simple_cnn.hdf5'

keras_model = load_model(MODEL_HDF5)
keras_model.summary()

keras2caffe.convert(keras_model, 'agegender.prototxt', 'agegender.caffemodel')

net  = caffe.Net('agegender.prototxt', 'agegender.caffemodel', caffe.TEST)

# ----------------------------------------------
# data
# ----------------------------------------------

img = cv2.imread('agegender/annotations/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg')
#img = cv2.imread('myself.jpg')
img = cv2.resize(img, (32, 32))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data /= 255

# ----------------------------------------------
# verify
# ----------------------------------------------

pred = keras_model.predict(data)[0]
prob = np.max(pred)
cls = pred.argmax()
lines=open('agegender_words.txt').readlines()
print prob, cls, lines[cls]

data = data.transpose((0, 3, 1, 2))

out = net.forward_all(data = data)
pred = out['predictions']
prob = np.max(pred)
cls = pred.argmax()
lines=open('agegender_words.txt').readlines()
print prob, cls, lines[cls]

