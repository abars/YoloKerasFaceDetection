#Generate annotation for keras
#https://susanqq.github.io/UTKFace/

from scipy import io as spio
from datetime import datetime

import numpy as np
import os
import shutil
import glob

if(os.path.exists("./dataset/UTKFace/")):
	DATASET_ROOT_PATH=""
else:
	DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

OUTPUT_LABEL="agegender_utk"

UTKFACE_PATH=DATASET_ROOT_PATH+"dataset/UTKFace/"
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
	if(gender==1):
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
for image_path in glob.glob(UTKFACE_PATH+"*.jpg"):
	image_name = os.path.basename(image_path)
	print image_name
	age, gender = image_name.split("_")[:2]
	gender=int(gender)
	age=min(int(age), 100)

	train_or_validation="train"
	if(i%4==0):
		train_or_validation="validation"

	src_img=UTKFACE_PATH+image_name
	shutil.copyfile(src_img, AGE_PATH+train_or_validation+"/"+get_age_path(age)+"/"+str(i)+".jpg")
	shutil.copyfile(src_img, GENDER_PATH+train_or_validation+"/"+get_gender_path(gender)+"/"+str(i)+".jpg")
	shutil.copyfile(src_img, AGE101_PATH+train_or_validation+"/"+format(age,"03d")+"/"+str(i)+".jpg")

	i=i+1
