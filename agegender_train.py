# ----------------------------------------------
# Train age gender classifier
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

# ----------------------------------------------
# MODE
# ----------------------------------------------

ANNOTATIONS=''
MODELS=''
DATASET_ROOT_PATH=""
AUGUMENTATION_MODE=""

#DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 3:
  ANNOTATIONS = sys.argv[1]
  MODELS = sys.argv[2]
  if len(sys.argv) >= 4:
    DATASET_ROOT_PATH=sys.argv[3]
  if len(sys.argv) >= 5:
    AUGUMENTATION_MODE=sys.argv[4]
else:
  print("usage: python agegender_train.py [agegender/gender/age/age101] [inceptionv3/vgg16/squeezenet/mobilenet] [datasetroot(optional)]")
  sys.exit(1)

if ANNOTATIONS!="agegender" and ANNOTATIONS!="gender" and ANNOTATIONS!="age" and ANNOTATIONS!="age101":
  print("unknown annotation mode");
  sys.exit(1)

if MODELS!="inceptionv3" and MODELS!="vgg16" and MODELS!="squeezenet" and MODELS!="mobilenet":
  print("unknown network mode");
  sys.exit(1)

if AUGUMENTATION_MODE!="" and AUGUMENTATION_MODE!="augumented":
  print("unknown augumentation mode");
  sys.exit(1)

# ----------------------------------------------
# Model
# ----------------------------------------------

if AUGUMENTATION_MODE=="":
  DATA_AUGUMENTATION=False
else:
  DATA_AUGUMENTATION=True

BATCH_SIZE = 32
EPOCS = 25

AUGUMENT=""
if DATA_AUGUMENTATION:
  AUGUMENT="augumented"

PLOT_FILE=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'_'+AUGUMENT+'.png'
MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+'_'+AUGUMENT+'.hdf5'

#Size
if ANNOTATIONS=='agegender':
  N_CATEGORIES = 16
if ANNOTATIONS=='age':
  N_CATEGORIES=8
if ANNOTATIONS=='gender':
  N_CATEGORIES=2
if ANNOTATIONS=='age101':
  N_CATEGORIES=101

#model
if(MODELS=='inceptionv3'):
   IMAGE_SIZE = 299
   input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
   base_model = InceptionV3(weights='imagenet', include_top=False,input_tensor=input_tensor)

   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(512, activation='relu')(x)
   predictions = Dense(N_CATEGORIES, activation='softmax')(x)

   model = Model(inputs=base_model.input, outputs=predictions)

   layer_num = len(model.layers)
   for layer in model.layers[:279]:
      layer.trainable = False
   for layer in model.layers[279:]:
      layer.trainable = True
elif(MODELS=='vgg16'):
   IMAGE_SIZE = 224
   input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
   base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(1024, activation='relu')(x)
   predictions = Dense(N_CATEGORIES, activation='softmax')(x)
   model = Model(inputs=base_model.input, outputs=predictions)
   for layer in base_model.layers[:15]:
      layer.trainable = False
elif(MODELS=='squeezenet'):
  IMAGE_SIZE=227
  import sys
  sys.path.append('../keras-squeezenet-master')
  from keras_squeezenet import SqueezeNet
  input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
  base_model = SqueezeNet(weights="imagenet", include_top=False, input_tensor=input_tensor)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(N_CATEGORIES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
elif(MODELS=='mobilenet'):
  IMAGE_SIZE=128
  input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)
  base_model = MobileNet(weights='imagenet', include_top=False,input_shape=input_shape)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(N_CATEGORIES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
else:
   raise Exception('invalid model name')

if(MODELS=='inceptionv3' or MODELS=='vgg16' or MODELS=='squeezenet' or MODELS=='mobilenet'):
  #for fine tuning
  from keras.optimizers import SGD
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
else:
  #for full training
  from keras.optimizers import Adagrad
  model.compile(optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

# ----------------------------------------------
# Data Augumentation
# ----------------------------------------------

# reference from https://github.com/yu4u/age-gender-estimation/blob/master/random_eraser.py
# https://github.com/yu4u/age-gender-estimation/blob/master/LICENSE
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img
    return eraser

# ----------------------------------------------
# Data
# ----------------------------------------------

preprocessing_function=None
if DATA_AUGUMENTATION:
  preprocessing_function=get_random_eraser(v_l=0, v_h=255)

train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10,
   preprocessing_function=preprocessing_function
)

test_datagen = ImageDataGenerator(
   rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
   DATASET_ROOT_PATH+'dataset/agegender/annotations/'+ANNOTATIONS+'/train',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   DATASET_ROOT_PATH+'dataset/agegender/annotations/'+ANNOTATIONS+'/validation',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

# ----------------------------------------------
# Train
# ----------------------------------------------

fit = model.fit_generator(train_generator,
   epochs=EPOCS,
   verbose=1,
   validation_data=validation_generator,
)

model.save(MODEL_HDF5)

# ----------------------------------------------
# Plot
# ----------------------------------------------

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="accuracy for training")
    axR.plot(fit.history['val_acc'],label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig(PLOT_FILE)
plt.close()
