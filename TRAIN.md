# How to train using Keras and Darknet

# Install

## Keras

`pip install keras`

## Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

# Face Detection (FDDB)

## Create dataset

Download fddb dataset (FDDB-folds and originalPics folder) and put in the dataset/fddb folder.

http://vis-www.cs.umass.edu/fddb/

Create datase/fddb/FDDB-folds/annotations_darknet folder for darknet.

`perl annotation_fddb_darknet.pl`

Preview converted annotations.

`python annotation_view.py fddb`

[![FDDB dataset overview](https://img.youtube.com/vi/KGeY_PFhRYA/0.jpg)](https://www.youtube.com/watch?v=KGeY_PFhRYA&feature=youtu.be)

## Train using Darknet

Here is a training using YoloV2.

`cd darknet`

`./darknet detector train data/face-one-class.data cfg/yolov2-tiny-train-one-class.cfg`

## Test using Darknet

Here is a test.

`./darknet detector demo data/face-one-class.data cfg/yolov2-tiny-train-one-class.cfg backup-face/yolov2-tiny-train-one-class_32600.weights -c 0`

## Training Result

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/log/yolov2-tiny-train-one-class_32600.jpg" width="50%" height="50%">

<http://www.abars.biz/keras/yolov2-tiny-one-class.cfg>

<http://www.abars.biz/keras/yolov2-tiny-train-one-class_32600.weights>

## Convert to Keras Model

Download YAD2K

https://github.com/allanzelener/YAD2K

This is a convert script.

`python3 yad2k.py yolov2-tiny-train-one-class.cfg yolov2-tiny-train-one-class_32600.weights yolov2_tiny-face.h5`

This is a converted model.

<http://www.abars.biz/keras/yolov2_tiny-face.h5>

# Age and Gender classification

## Create Dataset

### Use AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset (agegender folder)  and put in the dataset/agegender folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Create dataset/agegender/annotations for keras.

`perl annotation_agegender_keras.pl`

### Use IMDB-WIKI dataset

Download IMDB-WIKI dataset (Download faces only 7gb) and put in the dataset/imdb_crop folder.

https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Create dataset/agegender/annotations for keras.

`python annotation_imdb_keras.py`

Note : The age labels in the IMDB-WIKI dataset are noisy.

https://github.com/yu4u/age-gender-estimation/issues/15

### Use UTKFace dataset

Download UTKFace dataset and put in the dataset/imdb_crop folder.

https://susanqq.github.io/UTKFace/

Create dataset/agegender/annotations for keras.

`python annotation_utkface_keras.py`

## Train using Keras

Install keras-squeezenet

https://github.com/rcmalli/keras-squeezenet

Run classifier task using keras.

`python agegender_train.py age101 squeezenet`

`python agegender_train.py gender squeezenet`

## Test using Keras

Test classifier task using keras.

`python agegender_predict.py age101 squeezenet`

`python agegender_predict.py gender squeezenet`

## Training result

Age

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/agegender_age101_squeezenet.png" width="50%" height="50%">

<http://www.abars.biz/keras/agegender_age101_squeezenet.hdf5>

Gender

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/agegender_gender_squeezenet.png" width="50%" height="50%">

<http://www.abars.biz/keras/agegender_gender_squeezenet.hdf5>
