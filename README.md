# Yolo Keras Face Detection

Implement Face and Hand Detection , and Age and Gender Classification. (Experimental)

# Overview

## Functions

Face Detection (Keras , Darknet , Caffe)

Hand Detection (Darknet , Caffe)

Age Classification (Keras)

Gender Classification (Keras)

## Requirements

Keras2

Darknet

Caffe

Python 2.7

Perl

OpenCV

## Implement Environment

Mac Pro 2013

MacOS X 10.12

PlaidML

## Pretrained Model

### Age Classification

loss: 0.3099 - acc: 0.8952 - val_loss: 1.1309 - val_acc: 0.6939

<http://www.abars.biz/keras/train_age_vgg16.hdf5>

<http://www.abars.biz/keras/agegender_age_vgg16.prototxt>

<http://www.abars.biz/keras/agegender_age_vgg16.caffemodel>

### Gender Classification

<http://www.abars.biz/keras/train_gender_vgg16.hdf5>

<http://www.abars.biz/keras/agegender_gender_vgg16.prototxt>

<http://www.abars.biz/keras/agegender_gender_vgg16.caffemodel>

### Face Detection

from <https://github.com/dannyblueliu/YOLO-version-2-Face-detection>

<http://www.abars.biz/keras/face.prototxt>

<http://www.abars.biz/keras/face.caffemodel>

## Pretrained Model Demo

Here is a run using hdf5.

`python agegender_yolo.py keras`

Here is a run using caffemodel.

`python agegender_yolo.py converted`

## Reference Model Demo

Here is a run using reference model <https://gist.github.com/GilLevi/c9e99062283c719c03de>.

`python agegender_yolo.py caffe`

# Face Detection

## Create dataset

Download wider face dataset (wider_face_split and WIDER_train folder) and put in the same folder.

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

Create WIDER_train/annotations_keras folder for keras.

`perl annotation_widerface_keras.pl`

Create WIDER_train/annotations_darknet folder for darknet.

`perl annotation_widerface_darknet.pl`

## Train using Keras

Download BasicYoloKeras and put in the same folder.

https://github.com/experiencor/basic-yolo-keras

Here is a train.

`cd basic-yolo-keras-master`

`python train.py -c ../widerface_keras.json`

## Train using Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

Compile after modify src/yolo.c.

`char *train_images = "../WIDER_train/annotations_darknet/train.txt";`

`char *backup_directory = "backup/";`

`draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, l.classes);`

`int class = find_int_arg(argc, argv, "-class", 20);`

`else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, class, frame_skip, prefix, out_filename);`

Here is a train.

`cd darknet`

`./darknet yolo train ../widerface_tinyyolov1.cfg`

## Test using Darknet

Here is a test.

`./darknet yolo test ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights ../WIDER_train/annotations_darknet/1.jpg`

Here is a run.

`./darknet yolo demo ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights -class 1`

## Convert to CaffeModel

Download pytorch-caffe-darknet-convert and put in the same folder.

https://github.com/marvis/pytorch-caffe-darknet-convert

Convert to Caffe model.

`cd pytorch-caffe-darknet-convert`

`python darknet2caffe.py ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights widerface.prototxt widerface.caffemodel`

# Hand detection

## Create Dataset

Download viva hand dataset (detectiondata folder) and put in the same folder.

http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/

Create detectiondata/train/pos annotations for darknet.

`perl annotation_vivahand_darknet.pl`

## Train using Darknet

Compile with modify src/yolo.c.

`char *train_images = "../detectiondata/train/pos/train.txt";`

`char *backup_directory = "backup/";`

`draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, l.classes);`

`int class = find_int_arg(argc, argv, "-class", 20);`

`else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, class, frame_skip, prefix, out_filename);`

Here is a train.

`cd darknet`

`./darknet yolo train ../vivahand_tinyyolov1.cfg`

## Test using Darknet

Here is a test.

`./darknet yolo test ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights ../detectiondata/train/pos/1_0000003_0_0_0_6.png`

Here is a run.

`./darknet yolo demo ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights -class 4`

# Age and Gender classification

## Create Dataset

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset (agegender folder)  and put in the same folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Create agegender/ annotations for keras.

`perl annotation_agegender_keras.pl`

## Train using Keras

Run classifier task using keras.

`python agegender_train.py agegender`

`python agegender_train.py age`

`python agegender_train.py gender`

## Test using Keras

Test classifier task using keras.

`python agegender_predict.py agegender`

`python agegender_predict.py age`

`python agegender_predict.py gender`

Demo classifier task using keras and opencv.

`python agegender_recognize.py agegender`

`python agegender_recognize.py age`

`python agegender_recognize.py gender`

Demo classifier task using keras and yolo.

`python agegender_yolo.py keras`

# Related Work

<https://www.openu.ac.il/home/hassner/projects/cnn_agegender/>

<https://github.com/dpressel/rude-carnie>

<https://how-old.net/>
