# ----------------------------------------------
# Create hdf5 data from annotation data for training speed
# ----------------------------------------------

import os.path,sys
import numpy as np

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers import InputLayer
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D,Input
from keras.layers import SeparableConv2D
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import layers

import keras.callbacks

import matplotlib.pyplot as plt

ANNOTATIONS="gender"
DATASET_NAME="appareal"
DATASET_ROOT_PATH="c:/keras/"
OUTPUT_DATASET_ROOT_PATH="c:/keras/"

if len(sys.argv) >= 3:
  ANNOTATIONS = sys.argv[1]
  DATASET_NAME = sys.argv[2]
  if len(sys.argv) >= 4:
    DATASET_ROOT_PATH=sys.argv[3]
  if len(sys.argv) >= 5:
    OUTPUT_DATASET_ROOT_PATH=sys.argv[4]
else:
  print("usage: python annotation_to_hdf5.py [gender/age/age101] [adience/imdb/utk/appareal/vggface2/merged] [input_datasetroot(optional)] [output_datasetroot(optional)]")
  sys.exit(1)

if(os.path.exists("./dataset/"+DATASET_NAME)):
	DATASET_ROOT_PATH="./"
else:
	DATASET_ROOT_PATH="c:/keras/"

INPUT_PATH=DATASET_ROOT_PATH+"dataset/agegender_"+DATASET_NAME
OUTPUT_PATH=OUTPUT_DATASET_ROOT_PATH+"dataset/"+DATASET_NAME+'_'+ANNOTATIONS+'.h5'

IMAGE_SIZE=64

train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10,
   preprocessing_function=None
)

test_datagen = ImageDataGenerator(
   rescale=1.0 / 255
)

BATCH_SIZE=32

train_generator = train_datagen.flow_from_directory(
   INPUT_PATH+'/annotations/'+ANNOTATIONS+'/train',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   INPUT_PATH+'/annotations/'+ANNOTATIONS+'/validation',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

training_data_n = len(train_generator.filenames)
validation_data_n = len(validation_generator.filenames)

training_class_n=len(train_generator.class_indices)
validation_class_n=len(validation_generator.class_indices)

import h5py
f = h5py.File(OUTPUT_PATH, 'w')
train_x = f.create_dataset('training_x', (training_data_n,IMAGE_SIZE,IMAGE_SIZE,3), dtype='f')
train_y = f.create_dataset('training_y', (training_data_n,training_class_n), dtype='f')
validation_x = f.create_dataset('validation_x', (validation_data_n,IMAGE_SIZE,IMAGE_SIZE,3), dtype='f')
validation_y = f.create_dataset('validation_y', (validation_data_n,training_class_n), dtype='f')

cnt=0
for x_batch, y_batch in train_generator:
  for i in range(BATCH_SIZE):
    train_x[cnt] = x_batch[i]
    train_y[cnt] = y_batch[i]
    cnt = cnt+1
    if cnt>=training_data_n:
        break
  if cnt>=training_data_n:
    break

cnt=0
for x_batch, y_batch in validation_generator:
  for i in range(BATCH_SIZE):
    validation_x[cnt] = x_batch[i]
    validation_y[cnt] = y_batch[i]
    cnt = cnt+1
    if cnt>=validation_data_n:
        break
  if cnt>=validation_data_n:
    break

f.close()
