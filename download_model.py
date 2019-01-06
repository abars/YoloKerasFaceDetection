#Download pretrained model

import os
import sys
if sys.version_info >= (3,0):
	from urllib import request
else:
	import urllib2

def main(argv):
	OUTPUT_PATH="./pretrain/"
	if not os.path.isdir(OUTPUT_PATH):
		os.mkdir(OUTPUT_PATH)
	print("1/3");
	with open(OUTPUT_PATH+'agegender_age101_squeezenet.hdf5','wb') as f:
		path="https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/agegender_age101_squeezenet_imdb.hdf5"
		if sys.version_info >= (3,0):
			f.write(request.urlopen(path).read())
		else:
			f.write(urllib2.urlopen(path).read())
		f.close()
	print("2/3");
	with open(OUTPUT_PATH+'agegender_gender_squeezenet.hdf5','wb') as f:
		path="https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/agegender_gender_squeezenet_imdb.hdf5"
		if sys.version_info >= (3,0):
			f.write(request.urlopen(path).read())
		else:
			f.write(urllib2.urlopen(path).read())
		f.close()
	print("3/3");
	with open(OUTPUT_PATH+'yolov2_tiny-face.h5','wb') as f:
		path="https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/yolov2_tiny-face.h5"
		if sys.version_info >= (3,0):
			f.write(request.urlopen(path).read())
		else:
			f.write(urllib2.urlopen(path).read())
		f.close()

if __name__=='__main__':
	main(sys.argv[1:])
