# Yolo Keras Face Detection

Implement Face and Emotion detection , and Age and Gender Classification. (Experimental)

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/demo/demo.jpg" width="50%" height="50%">
(image from wider face dataset)

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

# Demo

## Pretrained Model

### Age and Gender Classification

<https://gist.github.com/GilLevi/c9e99062283c719c03de>

Download gender_net.caffemodel and gender_net.prototxt and age_net.caffemodel and age_net.prototxt.

Put in pretain folder.

### Face Detection

from <https://github.com/dannyblueliu/YOLO-version-2-Face-detection>

<http://www.abars.biz/keras/face.prototxt>

<http://www.abars.biz/keras/face.caffemodel>

Download face.prototxt and face.caffemodel.

Put in pretain folder.

### Emotion Detection

<https://gist.github.com/GilLevi/54aee1b8b0397721aa4b>

Download EmotiW_VGG_S.caffemodel and EmotiW_VGG_S.caffemodel.prototxt (rename from deploy.prototxt).

Put in pretain folder.

### Pretrained Model Demo

Here is a run using reference model .

`python agegender_demo.py caffe`

# How to train using Keras and Darknet

Here is a training tutorial. (Experimental)

https://github.com/abars/YoloKerasFaceDetection/blob/master/TRAIN.md

# Related Work

<https://www.openu.ac.il/home/hassner/projects/cnn_agegender/>

<https://github.com/dpressel/rude-carnie>

<https://how-old.net/>
