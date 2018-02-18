# How to train using Keras and Darknet

# Our Pretrained Model

## Age Classification

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/log/agegender_age_miniXception.png" width="50%" height="50%">

<http://www.abars.biz/keras/agegender_age_miniXception.hdf5>

<http://www.abars.biz/keras/agegender_age_miniXception.prototxt>

<http://www.abars.biz/keras/agegender_age_miniXception.caffemodel>

## Gender Classification

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/log/agegender_gender_simple_cnn.png" width="50%" height="50%">

<http://www.abars.biz/keras/agegender_gender_simple_cnn.hdf5>

## Face Detection

<http://www.abars.biz/keras/fddb_yolosmallv1_36000.weights>

<https://github.com/abars/YoloKerasFaceDetection/blob/master/cfg/fddb_yolosmallv1.cfg>

<!--
## Hand Detection

<http://www.abars.biz/keras/vivahand_tinyyolov1_19000.weights>

<https://github.com/abars/YoloKerasFaceDetection/blob/master/vivahand_tinyyolov1.cfg>
-->

## Our Pretrained Model Demo

Here is a run using hdf5.

`python agegender_demo.py keras`

Here is a run using caffemodel.

`python agegender_demo.py converted`

# Install

## Modify Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

Compile with <https://github.com/abars/YoloKerasFaceDetection/blob/master/darknet_custom/yolo.c> for custom classes and custom cfg.

`void train_yolo(char *cfgfile, char *weightfile,const char *train_images,const char *backup_directory)`

`draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, l.classes);`

`int class = find_int_arg(argc, argv, "-class", 20);`

`char *train_images = find_char_arg(argc, argv, "-train", 0);`

`char *backup_directory = find_char_arg(argc, argv, "-backup", 0);`

`else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights, train_images, backup_directory);`

`else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, class, frame_skip, prefix, out_filename);`

# Face Detection (FDDB)

## Create dataset

Download fddb dataset (FDDB-folds and originalPics folder) and put in the dataset/fddb folder.

http://vis-www.cs.umass.edu/fddb/

Create datase/fddb/FDDB-folds/annotations_darknet folder for darknet.

`perl annotation_fddb_darknet.pl`

Preview converted annotations.

`perl annotation_view.pl fddb`

[![FDDB dataset overview](https://img.youtube.com/vi/KGeY_PFhRYA/0.jpg)](https://www.youtube.com/watch?v=KGeY_PFhRYA&feature=youtu.be)

## Train using Darknet

`cd darknet`

`./darknet yolo train ../cfg/fddb_yolosmallv1.cfg -train ../dataset/fddb/FDDB-folds/annotations_darknet/train.txt -backup ./backup/ -class 1`

## Test using Darknet

Here is a test.

`./darknet yolo test ../cfg/fddb_yolosmallv1.cfg ./backup/fddb_yolosmallv1_36000.weights ../dataset/fddb/originalPics/2002/07/19/big/img_18 -class 1`

Here is a run.

`./darknet yolo demo ../cfg/fddb_yolosmallv1.cfg ./backup/fddb_yolosmallv1_36000.weights -class 1`

## Convert to CaffeModel

Download pytorch-caffe-darknet-convert and put in the same folder.

https://github.com/marvis/pytorch-caffe-darknet-convert

Convert to Caffe model.

`cd pytorch-caffe-darknet-convert`

`python darknet2caffe.py ../cfg/fddb_yolosmallv1.cfg ./backup/fddb_yolosmallv1_36000.weights face.prototxt face.caffemodel`

# Age and Gender classification

## Create Dataset

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset (agegender folder)  and put in the dataset/agegender folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Create dataset/agegender/annotations for keras.

`perl annotation_agegender_keras.pl`

## Train using Keras

Run classifier task using keras.

Network input is 64x64x3.

`python agegender_train.py age miniXception`

Network input is 48x48x3.

`python agegender_train.py gender simple_cnn`

## Test using Keras

Test classifier task using keras.

`python agegender_predict.py age miniXception`

`python agegender_predict.py gender simple_cnn`

Demo classifier task using keras and yolo.

`python agegender_demo.py keras`

# Emotion classification

## Create Dataset

Download FER2013 dataset.

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Train

Implementing.
