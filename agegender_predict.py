# ----------------------------------------------
# Predict age gender classifier
# ----------------------------------------------

import caffe
import cv2
import sys
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

#import plaidml.keras
#plaidml.keras.install_backend()

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model

import keras2caffe

# ----------------------------------------------
# MODE
# ----------------------------------------------

#ANNOTATIONS='agegender'
ANNOTATIONS='gender'
#ANNOTATION='age'
#ANNOTATION='emotion'

#MODELS="vgg16"
#MODELS="small_cnn"
MODELS="simple_cnn"
#MODELS="miniXception"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) == 3:
  ANNOTATIONS = sys.argv[1]
  MODELS = sys.argv[2]
else:
  print("usage: python agegender_predict.py [agegender/gender/age] [inceptionv3/vgg16/small_cnn/simple_cnn/miniXception]")
  sys.exit(1)

if ANNOTATIONS!="agegender" and ANNOTATIONS!="gender" and ANNOTATIONS!="age" and ANNOTATIONS!="emotion":
  print("unknown annotation mode");
  sys.exit(1)

if MODELS!="inceptionv3" and MODELS!="vgg16" and MODELS!="small_cnn" and MODELS!="simple_cnn" and MODELS!="miniXception":
  print("unknown network mode");
  sys.exit(1)

# ----------------------------------------------
# converting
# ----------------------------------------------

MODEL_HDF5='pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.hdf5'
ANNOTATION_WORDS='words/agegender_'+ANNOTATIONS+'_words.txt'

if(ANNOTATIONS=="emotion"):
	MODEL_HDF5='pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5'
	ANNOTATION_WORDS='words/emotion_words.txt'

IMAGE_SIZE = 32
if(MODELS=='simple_cnn'):
	IMAGE_SIZE = 48
if(MODELS=='miniXception'):
	IMAGE_SIZE = 64
if(MODELS=='vgg16'):
	IMAGE_SIZE = 224

keras_model = load_model(MODEL_HDF5)
keras_model.summary()

keras2caffe.convert(keras_model, 'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', 'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel')

net  = caffe.Net('pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', 'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel', caffe.TEST)

# ----------------------------------------------
# data
# ----------------------------------------------

img = cv2.imread('dataset/agegender/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg')
#img = cv2.imread('dataset/agegender/annotations/agegender/validation/11_15-20_f/landmark_aligned_face.290.11594063605_713764ddeb_o.jpg')

img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
if(ANNOTATIONS=='emotion'):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = np.expand_dims(img, axis=2)
else:
	img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data /= 255
if(ANNOTATIONS=='emotion'):
	data = data*2 - 1

# ----------------------------------------------
# verify
# ----------------------------------------------

pred = keras_model.predict(data)[0]
prob = np.max(pred)
cls = pred.argmax()
lines=open(ANNOTATION_WORDS).readlines()
print prob, cls, lines[cls]

data = data.transpose((0, 3, 1, 2))

out = net.forward_all(data = data)
pred = out[net.outputs[0]]

prob = np.max(pred)
cls = pred.argmax()
lines=open(ANNOTATION_WORDS).readlines()
print prob, cls, lines[cls]

