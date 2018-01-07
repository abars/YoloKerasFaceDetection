# ----------------------------------------------
# Predict age gender classifier
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
# MODE
# ----------------------------------------------

ANNOTATIONS='agegender'
#ANNOTATIONS='gender'
#ANNOTATION='age'

MODELS="vgg16"
#MODELS="small_cnn"
#MODELS="simple_cnn"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) == 2:
  ANNOTATIONS = sys.argv[1]
else:
  print("usage: python agegender_predict.py [agegender/gender/age]")
  sys.exit(1)

if ANNOTATIONS!="agegender" and ANNOTATIONS!="gender" and ANNOTATIONS!="age":
  print("unknown annotation mode");
  sys.exit(1)

# ----------------------------------------------
# converting
# ----------------------------------------------

MODEL_HDF5='train_'+ANNOTATIONS+'_'+MODELS+'.hdf5'
ANNOTATION_WORDS='agegender_'+ANNOTATIONS+'_words.txt'

IMAGE_SIZE = 32
if(MODELS=='simple_cnn'):
	IMAGE_SIZE = 64
if(MODELS=='vgg16'):
	IMAGE_SIZE = 224

keras_model = load_model(MODEL_HDF5)
keras_model.summary()

keras2caffe.convert(keras_model, 'agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', 'agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel')

net  = caffe.Net('agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', 'agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel', caffe.TEST)

# ----------------------------------------------
# data
# ----------------------------------------------

img = cv2.imread('agegender/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg')
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data /= 255

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
if(MODELS=='vgg16'):
	pred = out['dense_2']
else:
	pred = out['predictions']
prob = np.max(pred)
cls = pred.argmax()
lines=open(ANNOTATION_WORDS).readlines()
print prob, cls, lines[cls]

