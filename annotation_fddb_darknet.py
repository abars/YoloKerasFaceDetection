#Generate annotation for yolo
#http://vis-www.cs.umass.edu/fddb/

from scipy import io as spio
from datetime import datetime

import re
import numpy as np
import os
import shutil
import sys
import glob
import cv2

if(os.path.exists("./dataset/fddb/")):
	DATASET_ROOT_PATH="./"
else:
	DATASET_ROOT_PATH="/Volumes/ST5/keras/"

OUTPUT_LABEL="annotations_darknet"
OUTPUT_BASE_PATH=DATASET_ROOT_PATH+"dataset/fddb/FDDB-folds/"+OUTPUT_LABEL

TRAIN_PATH=OUTPUT_BASE_PATH+"/train.txt"
TEST_PATH=OUTPUT_BASE_PATH+"/test.txt"

if(not os.path.exists(OUTPUT_BASE_PATH)):
	os.mkdir(OUTPUT_BASE_PATH)

f_train=open(TRAIN_PATH,mode="w")
f_test=open(TEST_PATH,mode="w")

file_no=0

for list in range(1,11):
	list2=str(list)
	if list<10:
		list2="0"+str(list)
	path=DATASET_ROOT_PATH+"dataset/fddb/FDDB-folds/FDDB-fold-"+str(list2)+"-ellipseList.txt"
	lines=open(path).readlines()

	line_no=0

	while True:
		if line_no>=len(lines):
			break

		line=lines[line_no]
		line_no=line_no+1

		file_path=line.replace("\n","")
		image_path=DATASET_ROOT_PATH+"dataset/fddb/originalPics/"+file_path+".jpg"

		file_no=file_no+1

		image=cv2.imread(image_path)
		#print(image.shape)
		imagew=image.shape[1]
		imageh=image.shape[0]

		copy_path=OUTPUT_BASE_PATH+"/"+str(file_no)+".jpg"
		relative_path="../dataset/fddb/FDDB-folds/"+OUTPUT_LABEL+"/"+str(file_no)+".jpg"

		if file_no%4 == 0:
			f_train.write(relative_path+"\n")
		else:
			f_test.write(relative_path+"\n")

		shutil.copyfile(image_path, copy_path)
		
		line_n=int(lines[line_no])
		line_no=line_no+1

		annotation_path=OUTPUT_BASE_PATH+"/"+str(file_no)+".txt"
		f_annotation=open(annotation_path,mode="w")

		for i in range(line_n):
			line=lines[line_no]
			line_no=line_no+1
			#print(line)
			data=line.split(" ")
			major_axis_radius=float(data[0])
			minor_axis_radius=float(data[1])
			angle=float(data[2])
			center_x=float(data[3])
			center_y=float(data[4])
			
			x=center_x
			y=center_y

			w=minor_axis_radius*2
			h=major_axis_radius*2

			category=0
			x=1.0*x/imagew
			y=1.0*y/imageh
			w=1.0*w/imagew
			h=1.0*h/imageh

			if w>0 and h>0 and x-w/2>=0 and y-h/2>=0 and x+w/2<=1 and y+h/2<=1:
				f_annotation.write(str(category)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n")
			else:
				print("Invalid position removed "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
			
		f_annotation.close()

f_train.close()
f_test.close()
