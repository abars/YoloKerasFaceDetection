#Generate annotation for keras

from scipy import io as spio
from datetime import datetime

import numpy as np
import os
import shutil
import glob

if(os.path.exists("./dataset/appa-real-release")):
	DATASET_ROOT_PATH=""
else:
	DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

OUTPUT_LABEL="agegender_appareal"

GENDER_PATH=DATASET_ROOT_PATH+"dataset/"+OUTPUT_LABEL+"/annotations/gender/"
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

if(not os.path.exists(AGE101_PATH)):
	os.mkdir(AGE101_PATH)
	os.mkdir(AGE101_PATH+"train")
	os.mkdir(AGE101_PATH+"validation")
	for i in range(0,101):
		os.mkdir(AGE101_PATH+"train/"+format(i, '03d'))
		os.mkdir(AGE101_PATH+"validation/"+format(i, '03d'))

train_gender_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/allcategories_train.csv").readlines()
valid_gender_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/allcategories_valid.csv").readlines()
test_gender_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/allcategories_test.csv").readlines()

train_age_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/gt_avg_train.csv").readlines()
valid_age_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/gt_avg_valid.csv").readlines()
test_age_lines=open(DATASET_ROOT_PATH+"dataset/appa-real-release/gt_avg_test.csv").readlines()

age_label={}

for line in train_age_lines+valid_age_lines+test_age_lines:
	obj=line.split(",")
	path=obj[0]
	if(path=="file_name"):
		continue
	age=int(obj[4].strip())
	age_label[path]=age

gender_label={}

for line in train_gender_lines+valid_gender_lines+test_gender_lines:
	obj=line.split(",")
	path=obj[0]
	if(path=="file"):
		continue
	gender=obj[1].strip()
	if(gender=="male"):
		gender="m"
	else:
		gender="f"
	gender_label[path]=gender

i=0
for line in train_age_lines:
	obj=line.split(",")
	path=obj[0]
	if(path=="file_name"):
		continue
	gender=gender_label[path]
	age=age_label[path]
	image_path=DATASET_ROOT_PATH+"dataset/appa-real-release/train/"+path+"_face.jpg"
	shutil.copyfile(image_path, GENDER_PATH+"train/"+gender+"/"+str(i)+".jpg")
	shutil.copyfile(image_path, AGE101_PATH+"train/"+format(age,"03d")+"/"+str(i)+".jpg")
	i=i+1

i=0
for line in (valid_age_lines):
	obj=line.split(",")
	path=obj[0]
	if(path=="file_name"):
		continue
	gender=gender_label[path]
	age=age_label[path]
	image_path=DATASET_ROOT_PATH+"dataset/appa-real-release/valid/"+path+"_face.jpg"
	shutil.copyfile(image_path, GENDER_PATH+"validation/"+gender+"/"+str(i)+".jpg")
	shutil.copyfile(image_path, AGE101_PATH+"validation/"+format(age,"03d")+"/"+str(i)+".jpg")
	i=i+1
