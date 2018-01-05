# Yolo Keras Face Detection

Implement Face Detection using wider face dataset. (Experimental)

# Requirements

Keras2

Darknet

Python 2.7

# Prepare

Download wider face datase (wider_face_split and WIDER_train folder) and put in the same folder.

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

# Create dataset

Create WIDER_train/annotations_keras folder for keras.

`perl annotation_widerface_keras.pl`

Create WIDER_train/annotations_darknet folder for darknet.

`perl annotation_widerface_darknet.pl`

# Train using Keras

Download BasicYoloKeras and put in the same folder.

https://github.com/experiencor/basic-yolo-keras

Execute train.

`cd basic-yolo-keras-master`

`python train.py -c ../widerface_keras.json`

# Train using Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

Compile with modify src/yolo.c.

`char *train_images = "../WIDER_train/annotations_darknet/train.txt";`

`char *backup_directory = "backup/";`

`draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, l.classes);`

`int class = find_int_arg(argc, argv, "-class", 20);`

`else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, class, frame_skip, prefix, out_filename);`

Execute train.

`cd darknet`

`./darknet yolo train ../widerface_tinyyolov1.cfg`

Execute test.

`./darknet yolo test ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights ../WIDER_train/annotations_darknet/1.jpg`

Execute demo.

`./darknet yolo demo ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights -class 1`

Download pytorch-caffe-darknet-convert and put in the same folder.

https://github.com/marvis/pytorch-caffe-darknet-convert

Convert to Caffe model.

`cd pytorch-caffe-darknet-convert`

`python darknet2caffe.py ../widerface_tinyyolov1.cfg ./backup/widerface_tinyyolov1_200.weights widerface.prototxt widerface.caffemodel`

# Appendix : Train hand detection

Download viva hand dataset (detectiondata folder) and put in the same folder.

http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/

Create detectiondata/train/pos annotations for darknet.

`perl annotation_vivahand_darknet.pl`

Compile with modify src/yolo.c.

`char *train_images = "../detectiondata/train/pos/train.txt";`

`char *backup_directory = "backup/";`

`draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, l.classes);`

`int class = find_int_arg(argc, argv, "-class", 20);`

`else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, class, frame_skip, prefix, out_filename);`

Execute train.

`cd darknet`

`./darknet yolo train ../vivahand_tinyyolov1.cfg`

Execute test.

`./darknet yolo test ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights ../detectiondata/train/pos/1_0000003_0_0_0_6.png`

Execute demo.

`./darknet yolo demo ../vivahand_tinyyolov1.cfg ./backup/vivahand_tinyyolov1_4000.weights -class 4`

# Appendix : Train age and gender detection

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset (agegender folder)  and put in the same folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Create agegender/ annotations for keras.

`perl annotation_agegender_keras.pl`

Run classifier task using keras.

`python agegender_train.py`

Test classifier task using keras.

`python agegender_predict.py`


