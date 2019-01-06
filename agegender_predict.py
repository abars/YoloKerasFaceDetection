# ----------------------------------------------
# Predict age gender classifier
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# ----------------------------------------------
# MODE
# ----------------------------------------------

ANNOTATIONS=''
DATASET_NAME=''
MODELS=""
DATASET_ROOT_PATH=""
OPTIONAL_MODE=""
DATA_AUGUMENTATION=False

#DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 3:
  ANNOTATIONS = sys.argv[1]
  MODELS = sys.argv[2]
  if len(sys.argv) >= 4:
    DATASET_NAME=sys.argv[3]
  if len(sys.argv) >= 5:
    DATASET_ROOT_PATH=sys.argv[4]
  if len(sys.argv) >= 6:
    OPTIONAL_MODE=sys.argv[5]
else:
  print("usage: python agegender_predict.py [gender/age/age101/emotion] [inceptionv3/vgg16/squeezenet/octavio] [adience/imdb/utk/appareal/vggface2/empty] [datasetroot(optional)] [benchmark/caffemodel(optional)]")
  sys.exit(1)

if ANNOTATIONS!="gender" and ANNOTATIONS!="age" and ANNOTATIONS!="age101" and ANNOTATIONS!="emotion":
  print("unknown annotation mode");
  sys.exit(1)

if MODELS!="inceptionv3" and MODELS!="vgg16" and MODELS!="squeezenet" and MODELS!="mobilenet" and MODELS!="octavio":
  print("unknown network mode");
  sys.exit(1)

if DATASET_NAME!="adience" and DATASET_NAME!="imdb" and DATASET_NAME!="utk" and DATASET_NAME!="appareal" and DATASET_NAME!="vggface2" and DATASET_NAME!="empty":
  print("unknown dataset name");
  sys.exit(1)

if OPTIONAL_MODE!="" and OPTIONAL_MODE!="benchmark" and OPTIONAL_MODE!="caffemodel":
  print("unknown optional mode");
  sys.exit(1)

if DATASET_NAME=="empty":
	DATASET_NAME=""
else:
	DATASET_NAME='_'+DATASET_NAME

# ----------------------------------------------
# converting
# ----------------------------------------------

AUGUMENT=""
if(DATA_AUGUMENTATION):
  AUGUMENT="augumented"

MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+DATASET_NAME+AUGUMENT+'.hdf5'
ANNOTATION_WORDS='words/agegender_'+ANNOTATIONS+'_words.txt'

if(ANNOTATIONS=="emotion"):
	ANNOTATION_WORDS='words/emotion_words.txt'

if(MODELS=="octavio"):
	if(ANNOTATIONS=="emotion"):
		MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5'
	if(ANNOTATIONS=="gender"):
		MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/gender_mini_XCEPTION.21-0.95.hdf5'

if(MODELS=="mobilenet"):
	import keras
	from keras.utils.generic_utils import CustomObjectScope
	with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
		keras_model = load_model(MODEL_HDF5)
else:
	keras_model = load_model(MODEL_HDF5)
keras_model.summary()

# ----------------------------------------------
# convert to caffe model
# ----------------------------------------------

if OPTIONAL_MODE=="caffemodel":
	os.environ["GLOG_minloglevel"] = "2"
	import caffe
	import keras2caffe
	prototxt=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+DATASET_NAME+'.prototxt'
	caffemodel=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+DATASET_NAME+'.caffemodel'
	keras2caffe.convert(keras_model, prototxt, caffemodel)

# ----------------------------------------------
# Benchmark
# ----------------------------------------------

if OPTIONAL_MODE=="benchmark":
	BENCHMARK_DATASET_NAME="imdb"
	BENCHMARK_DATASET_TARGET="validation"
	BATCH_SIZE=64

	shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()

	disp_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
	   DATASET_ROOT_PATH+'dataset/agegender_'+BENCHMARK_DATASET_NAME+'/annotations/'+ANNOTATIONS+'/'+BENCHMARK_DATASET_TARGET,
	   target_size=(shape[1], shape[2]),
	   batch_size=BATCH_SIZE,
	   class_mode='categorical'
	)

	DISTRIBUTION_FILE=DATASET_ROOT_PATH+'pretrain/benchmark_'+ANNOTATIONS+"_"+MODELS+DATASET_NAME+'.png'

	fig = plt.figure()
	ax1 = fig.add_axes((0.1, 0.6, 0.8, 0.3))
	ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.3))
	ax1.tick_params(labelbottom="on")
	ax2.tick_params(labelleft="on")

	max_cnt=len(disp_generator.filenames)
	#max_cnt=10

	x=np.zeros((max_cnt))
	y=np.zeros((max_cnt))
	t=np.zeros((max_cnt))

	cnt=0
	heatmap=np.zeros((len(disp_generator.class_indices),len(disp_generator.class_indices)))
	for x_batch, y_batch in disp_generator:
		for i in range(BATCH_SIZE):
			x[cnt]=y_batch[i][0]
			t[cnt]=y_batch[i].argmax()

			data=x_batch[i]
			data.shape = (1,) + data.shape
			pred = keras_model.predict(data)[0]
			cls = pred.argmax()

			y[cnt]=cls

			heatmap[int(y[cnt]),int(t[cnt])]=heatmap[int(y[cnt]),int(t[cnt])]+1

			cnt=cnt+1
			print(""+str(cnt)+"/"+str(max_cnt)+" ground truth:"+str(y_batch[i].argmax())+" predicted:"+str(cls))
			if cnt>=max_cnt:
				break
		if cnt>=max_cnt:
			break

	ax1.pcolor(heatmap, cmap=plt.cm.Blues)
	if heatmap.shape[0]<=2:
		for y in range(heatmap.shape[0]):
			for x in range(heatmap.shape[1]):
				ax1.text(x + 0.5, y + 0.5, '%.4f' % heatmap[y, x],
					horizontalalignment='center',
					verticalalignment='center',
				)

	ax1.set_title('ground truth vs predicted '+ANNOTATIONS)
	ax1.set_xlabel(ANNOTATIONS+'(ground truth)')
	ax1.set_ylabel(ANNOTATIONS+'(predicted)')
	ax1.legend(loc='upper right')

	ax2.hist(t, bins=len(disp_generator.class_indices))
	ax2.set_title('distribution of ground truth '+ANNOTATIONS)
	ax2.set_xlabel(ANNOTATIONS+'(ground truth)')
	ax2.set_ylabel('count')
	ax2.legend(loc='upper right')
	
	fig.savefig(DISTRIBUTION_FILE)
	sys.exit(1)

# ----------------------------------------------
# Normal test
# ----------------------------------------------

if(os.path.exists("./dataset/agegender_adience/")):
	DATASET_PATH_ADIENCE=""
else:
	DATASET_PATH_ADIENCE="/Volumes/TB4/Keras/"

if(os.path.exists("./dataset/agegender_imdb/")):
	DATASET_PATH_IMDB=""
else:
	DATASET_PATH_IMDB="/Volumes/TB4/Keras/"

image_list=[
	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg',
	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/11_15-20_f/landmark_aligned_face.290.11594063605_713764ddeb_o.jpg',
	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/3_15-20_m/landmark_aligned_face.291.11593667615_2cb80d1c2a_o.jpg',
	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/f/26707.jpg',
	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/f/26761.jpg',
	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/m/181.jpg',
	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/m/83.jpg'
]

for image in image_list:
	if not os.path.exists(image):
		print(image+" not found")
		continue

	print(image)
	img = cv2.imread(image)

	shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
	img = cv2.resize(img, (shape[1], shape[2]))
	if(ANNOTATIONS=='emotion' or MODELS=='octavio'):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = np.expand_dims(img, axis=2)
	else:
		img = img[...,::-1]  #BGR 2 RGB

	data = np.array(img, dtype=np.float32)
	data.shape = (1,) + data.shape

	data = data / 255.0

	if(ANNOTATIONS=='emotion' or MODELS=='octavio'):
		data = data*2 - 1

	if(ANNOTATIONS=="age101"):
		lines=[]
		for i in range(0,101):
			lines.append("age."+str(i))
	else:
		lines=open(ANNOTATION_WORDS).readlines()

	pred = keras_model.predict(data)[0]
	prob = np.max(pred)
	cls = pred.argmax()
	print("keras:",prob, cls, lines[cls])

# ----------------------------------------------
# Test caffemodel
# ----------------------------------------------

	if OPTIONAL_MODE=="caffemodel":
		net  = caffe.Net(prototxt, caffemodel, caffe.TEST)
		data = data.transpose((0, 3, 1, 2))
		out = net.forward_all(data = data)
		pred = out[net.outputs[0]]
		prob = np.max(pred)
		cls = pred.argmax()
		print("caffe:",prob, cls, lines[cls])
