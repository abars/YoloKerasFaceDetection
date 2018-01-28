# ----------------------------------------------
# Train age gender classifier
# ----------------------------------------------

import os.path,sys

import plaidml.keras
plaidml.keras.install_backend()

from keras.layers.convolutional import Convolution2D
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
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks

import matplotlib.pyplot as plt

# ----------------------------------------------
# MODE
# ----------------------------------------------

ANNOTATIONS='agegender'
#ANNOTATIONS='gender'
#ANNOTATIONS='age'

#MODELS="inceptionv3"
MODELS="vgg16"
#MODELS="small_cnn"
#MODELS="simple_cnn"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) == 2:
  ANNOTATIONS = sys.argv[1]
else:
  print("usage: python agegender_train.py [agegender/gender/age]")
  sys.exit(1)

if ANNOTATIONS!="agegender" and ANNOTATIONS!="gender" and ANNOTATIONS!="age":
  print("unknown annotation mode");
  sys.exit(1)

# ----------------------------------------------
# Model
# ----------------------------------------------

NUM_TRAINING = 8634
NUM_VALIDATION = 2889
BATCH_SIZE = 16

PLOT_FILE='agegender_'+ANNOTATIONS+'_'+MODELS+'.png'
MODEL_HDF5='agegender_'+ANNOTATIONS+'_'+MODELS+'.hdf5'

#Size
if ANNOTATIONS=='agegender':
  N_CATEGORIES = 16
if ANNOTATIONS=='age':
  N_CATEGORIES=8
if ANNOTATIONS=='gender':
  N_CATEGORIES=2

#model
if(MODELS=='inceptionv3'):
   IMAGE_SIZE = 299
   EPOCS = 50
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
   EPOCS = 50
   input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
   base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(1024, activation='relu')(x)
   predictions = Dense(N_CATEGORIES, activation='softmax')(x)
   model = Model(inputs=base_model.input, outputs=predictions)
   for layer in base_model.layers[:15]:
      layer.trainable = False
elif(MODEL_HDF5=='small_cnn'):
   IMAGE_SIZE = 32
   EPOCS = 50
   model = Sequential()
   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
   model.add(InputLayer(input_shape=input_shape))
   model.add(Convolution2D(96, 3, 3, border_mode='same'))
   model.add(Activation('relu'))
   model.add(Convolution2D(128, 3, 3))
   model.add(Activation('relu'))
   model.add(Dropout(0.5))
   model.add(Flatten())
   model.add(Dense(1024))
   model.add(Activation('relu'))
   model.add(Dropout(0.5))
   model.add(Dense(N_CATEGORIES))
   model.add(Activation('softmax',name='predictions'))
elif(MODEL_HDF5=='simple_cnn'):
   IMAGE_SIZE = 64
   EPOCS = 50
   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

   model = Sequential()
   model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                         name='image_array', input_shape=input_shape))
   model.add(BatchNormalization())
   model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
   model.add(BatchNormalization())
   model.add(Activation('relu'))
   model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
   model.add(Dropout(.5))

   model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
   model.add(BatchNormalization())
   model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
   model.add(BatchNormalization())
   model.add(Activation('relu'))
   model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
   model.add(Dropout(.5))

   model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
   model.add(BatchNormalization())
   model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
   model.add(BatchNormalization())
   model.add(Activation('relu'))
   model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
   model.add(Dropout(.5))

   model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
   model.add(BatchNormalization())
   model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
   model.add(BatchNormalization())
   model.add(Activation('relu'))
   model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
   model.add(Dropout(.5))

   model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
   model.add(BatchNormalization())
   model.add(Convolution2D(filters=N_CATEGORIES, kernel_size=(3, 3), padding='same'))
   model.add(GlobalAveragePooling2D())
   model.add(Activation('softmax',name='predictions'))
else:
   raise Exception('invalid model name')

#for fine tuning
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

#for full training
#from keras.optimizers import Adagrad
#model.compile(optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

# ----------------------------------------------
# Data
# ----------------------------------------------

#def preprocess_input(img):
  #img = img[...,::-1]  #RGB2BGR
  #img = img - (104,117,123) #BGR mean value of VGG16

  #img = img - (123,117,104) #RGB mean value of VGG16
  #return img

train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   #preprocessing_function=preprocess_input,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10)

test_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   #preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
   'dataset/agegender/annotations/'+ANNOTATIONS+'/train',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   'dataset/agegender/annotations/'+ANNOTATIONS+'/validation',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

# ----------------------------------------------
# Train
# ----------------------------------------------

fit = model.fit_generator(train_generator,
   steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
   epochs=EPOCS,
   verbose=1,
   validation_data=validation_generator,
   validation_steps=NUM_VALIDATION//BATCH_SIZE,
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
    axR.plot(fit.history['acc'],label="loss for training")
    axR.plot(fit.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig(PLOT_FILE)
plt.close()