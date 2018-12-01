# ----------------------------------------------
# Predict age gender classifier
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model

# ----------------------------------------------
# MODE
# ----------------------------------------------

#ANNOTATIONS='agegender'
ANNOTATIONS='gender'
#ANNOTATION='age'
#ANNOTATION='emotion'

#MODELS="vgg16"
#MODELS="small_cnn"
#MODELS="simple_cnn"
#MODELS="miniXception"
MODELS="squeezenet"
#MODELS="mobilenet"

DATASET_ROOT_PATH=""
#DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 3:
  ANNOTATIONS = sys.argv[1]
  MODELS = sys.argv[2]
  if len(sys.argv) >= 4:
    DATASET_ROOT_PATH=sys.argv[3]
else:
  print("usage: python agegender_predict.py [agegender/gender/age/age101/emotion] [inceptionv3/vgg16/small_cnn/simple_cnn/miniXception/squeezenet/octavio] [datasetroot(optional)]")
  sys.exit(1)

if ANNOTATIONS!="agegender" and ANNOTATIONS!="gender" and ANNOTATIONS!="age" and ANNOTATIONS!="age101" and ANNOTATIONS!="emotion":
  print("unknown annotation mode");
  sys.exit(1)

if MODELS!="inceptionv3" and MODELS!="vgg16" and MODELS!="small_cnn" and MODELS!="simple_cnn" and MODELS!="miniXception" and MODELS!="squeezenet" and MODELS!="mobilenet" and MODELS!="octavio":
  print("unknown network mode");
  sys.exit(1)

# ----------------------------------------------
# converting
# ----------------------------------------------

MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.hdf5'
ANNOTATION_WORDS='words/agegender_'+ANNOTATIONS+'_words.txt'

if(ANNOTATIONS=="emotion"):
	MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5'
	ANNOTATION_WORDS='words/emotion_words.txt'

if(MODELS=="octavio"):
	MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/gender_mini_XCEPTION.21-0.95.hdf5'
	ANNOTATION_WORDS='words/agegender_gender_words.txt'

if(MODELS=="mobilenet"):
	import keras
	from keras.utils.generic_utils import CustomObjectScope
	with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
		keras_model = load_model(MODEL_HDF5)
else:
	keras_model = load_model(MODEL_HDF5)
keras_model.summary()

shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
IMAGE_SIZE = shape[1]

# ----------------------------------------------
# test
# ----------------------------------------------

image_list=[
'dataset/agegender/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg',
'dataset/agegender/annotations/agegender/validation/11_15-20_f/landmark_aligned_face.290.11594063605_713764ddeb_o.jpg',
'dataset/agegender/annotations/agegender/validation/3_15-20_m/landmark_aligned_face.291.11593667615_2cb80d1c2a_o.jpg',
'/Volumes/TB4/Keras/dataset/agegender/annotations/gender/train/f/26707.jpg',
'/Volumes/TB4/Keras/dataset/agegender/annotations/gender/train/f/26761.jpg',
'/Volumes/TB4/Keras/dataset/agegender/annotations/gender/train/m/181.jpg',
'/Volumes/TB4/Keras/dataset/agegender/annotations/gender/train/m/83.jpg'
]

for image in image_list:
	if not os.path.exists(image):
		print image+" not found"
		continue

	print image
	img = cv2.imread(image)

	img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
	if(ANNOTATIONS=='emotion' or MODELS=='octavio'):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = np.expand_dims(img, axis=2)
	else:
		img = img[...,::-1]  #BGR 2 RGB

	data = np.array(img, dtype=np.float32)
	data.shape = (1,) + data.shape

	data = data / 255.0

	if(ANNOTATIONS=='emotion' or MODELS=='octavio'):
		data = data*2 - 1

	if(ANNOTATIONS=="age101"):
		lines=[]
		for i in range(0,101):
			lines.append("age."+str(i))
	else:
		lines=open(ANNOTATION_WORDS).readlines()

	pred = keras_model.predict(data)[0]
	print pred
	prob = np.max(pred)
	cls = pred.argmax()
	print prob, cls, lines[cls]

# ----------------------------------------------
# convert to caffe model
# ----------------------------------------------

sys.exit(1)

import keras2caffe
import caffe

keras2caffe.convert(keras_model, DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel')
net  = caffe.Net(DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.prototxt', DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'.caffemodel', caffe.TEST)

data = data.transpose((0, 3, 1, 2))

out = net.forward_all(data = data)
pred = out[net.outputs[0]]

prob = np.max(pred)
cls = pred.argmax()
print prob, cls, lines[cls]

