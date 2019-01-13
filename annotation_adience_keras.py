#Generate annotation for keras
#https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

from scipy import io as spio
from datetime import datetime

import re
import numpy as np
import os
import shutil
import sys
import glob

if(os.path.exists("./dataset/imdb_crop/")):
	DATASET_ROOT_PATH=""
else:
	DATASET_ROOT_PATH="/Volumes/ST5/Keras/"

OUTPUT_LABEL="agegender_adience"

ANNOTATION_FILES=DATASET_ROOT_PATH+"dataset/adience/fold_"
FACE_FILES=DATASET_ROOT_PATH+"dataset/adience/aligned"

GENDER_PATH=DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations/gender/"
AGE_PATH=DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations/age/"
AGE101_PATH=DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations/age101/"

if(not os.path.exists(DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/")):
	os.mkdir(DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL)
if(not os.path.exists(DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations/")):
	os.mkdir(DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations")

if(not os.path.exists(GENDER_PATH)):
	os.mkdir(GENDER_PATH)
	os.mkdir(GENDER_PATH+"train")
	os.mkdir(GENDER_PATH+"train/f")
	os.mkdir(GENDER_PATH+"train/m")
	os.mkdir(GENDER_PATH+"validation")
	os.mkdir(GENDER_PATH+"validation/f")
	os.mkdir(GENDER_PATH+"validation/m")

if(not os.path.exists(AGE_PATH)):
	os.mkdir(AGE_PATH)
	os.mkdir(AGE_PATH+"train")
	os.mkdir(AGE_PATH+"train/0-2")
	os.mkdir(AGE_PATH+"train/4-6")
	os.mkdir(AGE_PATH+"train/8-13")
	os.mkdir(AGE_PATH+"train/15-20")
	os.mkdir(AGE_PATH+"train/25-32")
	os.mkdir(AGE_PATH+"train/38-43")
	os.mkdir(AGE_PATH+"train/48-53")
	os.mkdir(AGE_PATH+"train/60-")
	os.mkdir(AGE_PATH+"validation")
	os.mkdir(AGE_PATH+"validation/0-2")
	os.mkdir(AGE_PATH+"validation/4-6")
	os.mkdir(AGE_PATH+"validation/8-13")
	os.mkdir(AGE_PATH+"validation/15-20")
	os.mkdir(AGE_PATH+"validation/25-32")
	os.mkdir(AGE_PATH+"validation/38-43")
	os.mkdir(AGE_PATH+"validation/48-53")
	os.mkdir(AGE_PATH+"validation/60-")

if(not os.path.exists(AGE101_PATH)):
	os.mkdir(AGE101_PATH)
	os.mkdir(AGE101_PATH+"train")
	os.mkdir(AGE101_PATH+"validation")
	for i in range(0,101):
		os.mkdir(AGE101_PATH+"train/"+format(i, '03d'))
		os.mkdir(AGE101_PATH+"validation/"+format(i, '03d'))

def get_gender_path(gender):
	if(gender=="f"):
		return "f"
	return "m"

def get_age_path(age):
	if(age>=0 and age<=3):
		return "0-2"
	if(age>=4 and age<=7):
		return "4-6"
	if(age>=8 and age<=14):
		return "8-13"
	if(age>=15 and age<=24):
		return "15-20"
	if(age>=25 and age<=37):
		return "25-32"
	if(age>=38 and age<=47):
		return "38-43"
	if(age>=48 and age<=59):
		return "48-53"
	return "60-"

i=0

for list in range(5):
	path=ANNOTATION_FILES+str(list)+"_data.txt"
	lines=open(path).readlines()
	for line in lines:
		data=line.split("\t");
		user_id,original_image,face_id,age,gender,x,y,dx,dy,tilt_ang,fiducial_yaw_angle,fiducial_score=data
		
		if age=="age":
			continue

		result = re.match(r"^[0-9]+$",age)
		if result:
			age_int=int(age)
		else:
			result = re.match(r"\(([0-9]+),",age)
			if result:
				age_int=int(result.group(1))
			else:
				age_int=-1

		age=age_int

		thumb_dir=FACE_FILES+"/"+user_id+"/";
		full_path=""
		for image_path in glob.glob(thumb_dir+"*"):
			image_name = os.path.basename(image_path)
			if image_name.find(face_id+"."+original_image)!=-1:
				full_path=image_path
				break

		if full_path=="":
			print("path not found")
			sys.exit(1)

		print("path:"+str(full_path)+" gender:"+str(gender))
		print(""+get_age_path(age)+" "+get_gender_path(gender))
		train_or_validation="train"
		if(i%4==0):
			train_or_validation="validation"
		src_img=full_path
		if age_int!=-1:
			shutil.copyfile(src_img, AGE_PATH+train_or_validation+"/"+get_age_path(age)+"/"+str(i)+".jpg")
		shutil.copyfile(src_img, GENDER_PATH+train_or_validation+"/"+get_gender_path(gender)+"/"+str(i)+".jpg")
		if age_int!=-1:
			shutil.copyfile(src_img, AGE101_PATH+train_or_validation+"/"+format(age,"03d")+"/"+str(i)+".jpg")

		i=i+1