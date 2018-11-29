# ----------------------------------------------
# Annotation View
# ----------------------------------------------

import os
import cv2
import sys
import numpy as np

def view(MODE):
	cnt=1

	if MODE == "fddb":
		datapath='dataset/fddb/FDDB-folds/annotations_darknet/'
	if MODE == "widerface":
		datapath='dataset/widerface/WIDER_train/annotations_darknet/'
	if MODE == "vivahand":
		datapath='dataset/vivahand/detectiondata/train/pos/'

	files=open(datapath+'train.txt').readlines()

	for file in files:
		file = file.replace('\n','')
		file = file.replace('\r','')

		path="dataset/"+file
		target_image = cv2.imread(path)
		path=path.replace(".jpg",".txt")
		path=path.replace(".png",".txt")

		lines=open(path).readlines()
		for line in lines:
			data=line.split(" ")

			cls=int(data[0])
			x=int(float(data[1])*target_image.shape[1])
			y=int(float(data[2])*target_image.shape[0])
			w=int(float(data[3])*target_image.shape[1])
			h=int(float(data[4])*target_image.shape[0])

			cv2.rectangle(target_image, (x-w/2,y-h/2), (x+w/2,y+h/2), color=(0,0,255), thickness=3)
			cv2.putText(target_image, str(cls), (x,y+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));

		cv2.imshow("agegender", target_image)

		k = cv2.waitKey(1)
		if k == 27:
			break

		cnt=cnt+1

	cv2.destroyAllWindows()

def main(argv):
	MODE="vivahand"
	if len(sys.argv) == 2:
		MODE = sys.argv[1]
	else:
		print("usage: python annotation_view.py [vivahand/widerface/fddb]")
		sys.exit(1)
	if(MODE!="vivahand" and MODE!="widerface" and MODE!="fddb"):
		print("Unknown mode "+MODE)
		sys.exit(1)
	view(MODE)

if __name__=='__main__':
	main(sys.argv[1:])

