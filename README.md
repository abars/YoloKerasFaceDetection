# Yolo Keras Face Detection

WiderFaceDataSetとBasicYoloKerasを使用して顔検出を行います。（実験中）

# 準備

WIDER FACE Datasetをダウンロードし、wider_face_splitフォルダとWIDER_trainフォルダをannotation.plと同じフォルダに置いて下さい。

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
# データセット作成

以下のコマンドでWIDER_train/annotations_kerasフォルダを作成します。

`perl annotation_keras.pl`

以下のコマンドでWIDER_train/annotations_darknetフォルダを作成します。

`perl annotation_darknet.pl`

# Kerasでの学習

Basic Yolo Kerasをダウンロードします。

https://github.com/experiencor/basic-yolo-keras

basic-yolo-keras-masterフォルダをwiderface.jsonと同じフォルダに置きます。

basic-yolo-keras-masterフォルダの中で以下のコマンドで学習します。

`python train.py -c ../widerface_keras.json`

# Darknetでの学習

Darknetをダウンロードします。Yolo v1で学習します。

https://github.com/pjreddie/darknet

src/yolo.cを書き換えてmakeします。

`char *train_images = "../WIDER_train/annotations_darknet/train.txt";`
`char *backup_directory = "backup/";`

学習します。

`./darknet yolo train ../widerface_tinyyolov1.cfg`

テストします。

`./darknet yolo test ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights ../WIDER_train/annotations_darknet/1.jpg`

# CaffeModelへの変換

CaffeYoloをダウンロードします。

https://github.com/xingwangsfu/caffe-yolo

モデルを変換します。

`python create_yolo_caffemodel.py -m yolo_train_val.prototxt -w yolo.weights -o yolo.caffemodel`

