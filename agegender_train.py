#Train anime face

import os.path,sys

#sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../../')
#os.environ['KERAS_BACKEND'] = 'theano'

import plaidml.keras
plaidml.keras.install_backend()

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers import InputLayer
from keras.models import Sequential

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks

N_CATEGORIES  = 16
BATCH_SIZE = 16

NUM_TRAINING = 14490*3/4
NUM_VALIDATION = 14490*1/4

#MODEL_HDF5='train_vgg16.hdf5'
MODEL_HDF5='train_small_cnn2.hdf5'
#MODEL_HDF5='train_small_cnn.hdf5'

#VOC model
if(MODEL_HDF5=='train_vgg16'):
   IMAGE_SIZE = 224
   input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
   base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(1024, activation='relu')(x)
   predictions = Dense(N_CATEGORIES, activation='softmax')(x)
   model = Model(inputs=base_model.input, outputs=predictions)
   #for layer in base_model.layers:
   #   layer.trainable = False
   for layer in base_model.layers[:15]:
      layer.trainable = False
elif(MODEL_HDF5=='train_small_cnn2.hdf5'):
   IMAGE_SIZE = 32
   model = Sequential()
   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
   model.add(InputLayer(input_shape=input_shape))
   model.add(Convolution2D(32, kernel_size=(3, 3)))
   model.add(Activation('relu'))
   model.add(Convolution2D(64, (3, 3)))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))
   model.add(Flatten())
   model.add(Dense(128))
   model.add(Activation('relu'))
   model.add(Dropout(0.5))
   model.add(Dense(N_CATEGORIES))
   model.add(Activation('softmax'))
elif(MODEL_HDF5=='train_small_cnn.hdf5'):
   IMAGE_SIZE = 32
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
   model.add(Activation('softmax'))
else:
   raise Exception('invalid model name')

#from keras.optimizers import SGD
  #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

from keras.optimizers import Adagrad
model.compile(optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10)

test_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
)

train_generator = train_datagen.flow_from_directory(
   'agegender/annotations/train',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   'agegender/annotations/validation',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

hist = model.fit_generator(train_generator,
   steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
   epochs=50,
   verbose=1,
   validation_data=validation_generator,
   validation_steps=NUM_VALIDATION//BATCH_SIZE,
   )

model.save(MODEL_HDF5)