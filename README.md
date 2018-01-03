# Yolo Keras Face Detection

Implement Face Detection using wider face dataset.(Experimental)

# Requirements

Keras2

Python 2.7

# Prepare

Download wider face datase (wider_face_split and WIDER_train folder) and put in the same folder.

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

# Create dataset

Create WIDER_train/annotations_keras folder for keras.

`perl annotation_keras.pl`

Create WIDER_train/annotations_darknet folder for darknet.

`perl annotation_darknet.pl`

# Train using Keras

Download BasicYoloKeras and put in the same folder.

https://github.com/experiencor/basic-yolo-keras

Execute train.

`cd basic-yolo-keras-master`
`python train.py -c ../widerface_keras.json`

# Train using Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

Execute train.

`char *train_images = "../WIDER_train/annotations_darknet/train.txt";`
`char *backup_directory = "backup/";`

Execute train.

`./darknet yolo train ../widerface_tinyyolov1.cfg`

Execute test.

`./darknet yolo test ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights ../WIDER_train/annotations_darknet/1.jpg`

Download CaffeYolo and put in the same folder.

https://github.com/xingwangsfu/caffe-yolo

Convert to Caffe model.

`python create_yolo_caffemodel.py -m yolo_train_val.prototxt -w yolo.weights -o yolo.caffemodel`

