#Generate annotation for keras

from scipy import io as spio
from datetime import datetime

import numpy as np
import os
import shutil
import glob

if(os.path.exists("./dataset/vggface2/identity_meta_with_estimated_age.csv")):
	DATASET_ROOT_PATH=""
else:
	DATASET_ROOT_PATH="/Volumes/ST5/keras/"

OUTPUT_LABEL="agegender_vggface2"

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

lines=open(DATASET_ROOT_PATH+"dataset/vggface2/identity_meta_with_estimated_age.csv").readlines()

i=0

for line in lines:
	obj=line.split(", ")
	path=obj[0]
	trainset=obj[3]
	gender=obj[4]
	age=int(obj[5].strip())

	if trainset=="1":
		train_or_validation="train"
	else:
		train_or_validation="validation"

	print(line.strip(),train_or_validation)

	if trainset=="0":
		path2=DATASET_ROOT_PATH+"Dataset/vggface2/test/"+path
	else:
		path2=DATASET_ROOT_PATH+"Dataset/vggface2/train/"+path

	for image_path in glob.glob(path2+"/*.jpg"):
		shutil.copyfile(image_path, GENDER_PATH+train_or_validation+"/"+gender+"/"+str(i)+".jpg")
		shutil.copyfile(image_path, AGE101_PATH+train_or_validation+"/"+format(age,"03d")+"/"+str(i)+".jpg")
		i=i+1

