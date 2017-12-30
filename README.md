# Yolo Face Detection

WiderFaceDataSetとBasicYoloKerasを使用して顔検出を行います。（実験中）

# 準備

WIDER FACE Datasetをダウンロードし、wider_face_splitフォルダとWIDER_trainフォルダをannotation.plと同じフォルダに置いて下さい。

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
# データセット作成

以下のコマンドでWIDER_train/annotationsフォルダを作成します。

`perl annotation.pl`

# 学習

Basic Yolo Kerasをダウンロードします。

https://github.com/experiencor/basic-yolo-keras

basic-yolo-keras-masterフォルダをwiderface.jsonと同じフォルダに置きます。

basic-yolo-keras-masterフォルダの中で以下のコマンドで学習します。

`python train.py -c ../widerface.json`
