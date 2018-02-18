# Experimental Training

# Face Detection (Widerface)

## Create dataset

Download wider face dataset (wider_face_split and WIDER_train folder) and put in the dataset/widerface folder.

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

Create dataset/widerface/WIDER_train/annotations_keras folder for keras.

`perl annotation_widerface_keras.pl`

Create datase/widerface/WIDER_train/annotations_darknet folder for darknet.

`perl annotation_widerface_darknet.pl`

## Train using Darknet

Here is a train.

`cd darknet`

`./darknet yolo train ../cfg/widerface_tinyyolov1.cfg -train ../dataset/widerface/WIDER_train/annotations_darknet/train.txt -backup ./backup/ -class 1`

## Train using Keras

Download BasicYoloKeras and put in the same folder.

https://github.com/experiencor/basic-yolo-keras

Here is a train.

`cd basic-yolo-keras-master`

`python train.py -c ../cfg/widerface_keras.json`

##  Test using Darknet

Here is a test.

`./darknet yolo test ../cfg/widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_4000.weights ../dataset/widerface/WIDER_train/annotations_darknet/1.jpg -class 1`

Here is a run.

`./darknet yolo demo ../cfg/widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_4000.weights -class 1`

# Hand detection

## Create Dataset

Download viva hand dataset (detectiondata folder) and put in the datase/vivahand folder.

http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/

Create dataset/vivahand/detectiondata/train/pos annotations for darknet.

`perl annotation_vivahand_darknet.pl`

## Train using Darknet

Here is a train.

`cd darknet`

`./darknet yolo train ../cfg/vivahand_tinyyolov1.cfg -train ../dataset/vivahand/detectiondata/train/pos/train.txt -backup ./backup/ -class 4`

## Test using Darknet

Here is a test.

`./darknet yolo test ../cfg/vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights ../dataset/vivahand/detectiondata/train/pos/1_0000003_0_0_0_6.png -class 4`

Here is a run.

`./darknet yolo demo ../cfg/vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights -class 4`

