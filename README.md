# Yolo Keras Face Detection

Implement Face detection, and Age and Gender Classification, and Emotion Classification.

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/demo/demo.jpg" width="50%" height="50%">
(image from wider face dataset)

# Overview

## Functions

Face Detection (Darknet , Caffe)

Age Classification (Keras)

Gender Classification (Keras)

Emotion Classification (Keras)

## Requirements

Keras2 + Tensorflow

Darknet

Caffe

OpenCV

Python 2.7

Perl

## Implement Environment

Mac Pro 2013 (Xeon)

MacOS X 10.12

# Demo

## Pretrained Model

### Age and Gender Classification

<https://gist.github.com/GilLevi/c9e99062283c719c03de>

Download gender_net.caffemodel, gender_net.prototxt, age_net.caffemodel and age_net.prototxt.

Put in pretain folder.

### Face Detection

Converted from <https://github.com/dannyblueliu/YOLO-version-2-Face-detection>

<http://www.abars.biz/keras/face.prototxt>

<http://www.abars.biz/keras/face.caffemodel>

Download face.prototxt and face.caffemodel.

Put in pretain folder.

### Emotion Detection

Converted from <https://github.com/oarriaga/face_classification>

<http://www.abars.biz/keras/emotion_miniXception.prototxt>

<http://www.abars.biz/keras/emotion_miniXception.caffemodel>

Download emotion_miniXception.prototxt and emotion_miniXception.caffemodel.

Put in pretain folder.

### Pretrained Model Demo

Here is a run using reference model .

`python agegender_demo.py caffe`

# How to train using Keras and Darknet

Here is a training tutorial. (Experimental)

https://github.com/abars/YoloKerasFaceDetection/blob/master/TRAIN.md

# Related Work

<https://github.com/oarriaga/face_classification>

<https://gist.github.com/GilLevi/54aee1b8b0397721aa4b>

<https://www.openu.ac.il/home/hassner/projects/cnn_agegender/>

<https://github.com/dpressel/rude-carnie>

<https://how-old.net/>
