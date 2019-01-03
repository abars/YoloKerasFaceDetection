#Generate annotation for keras
#https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

from scipy import io as spio
from datetime import datetime

import numpy as np
import os
import shutil

if(os.path.exists("./dataset/imdb_crop/")):
	DATASET_ROOT_PATH=""
else:
	DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

OUTPUT_LABEL="agegender_imdb"

IMDB_PATH=DATASET_ROOT_PATH+"dataset/imdb_crop/"
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

def calc_age(taken, dob):
	birth = datetime.fromordinal(max(int(dob) - 366, 1))
	if birth.month < 7:
		return taken - birth.year
	else:
		return taken - birth.year - 1

def is_valid(face_score,second_face_score,age,gender):
	if face_score < 1.0:
		return False
	if (~np.isnan(second_face_score)) and second_face_score > 0.0:
		return False
	if ~(0 <= age <= 100):
		return False
	if np.isnan(gender):
		return False
	return True

def get_gender_path(gender):
	if(gender==0.0):
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

meta = spio.loadmat(IMDB_PATH+"imdb.mat")

db = "imdb"

full_path = meta[db][0, 0]["full_path"][0]
dob = meta[db][0, 0]["dob"][0]
gender = meta[db][0, 0]["gender"][0]
photo_taken = meta[db][0, 0]["photo_taken"][0]
face_score = meta[db][0, 0]["face_score"][0]
second_face_score = meta[db][0, 0]["second_face_score"][0]
age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

for i in range(len(full_path)):
	if(not is_valid(face_score[i],second_face_score[i],age[i],gender[i])):
		continue
	print "path:"+str(full_path[i])+" gender:"+str(gender[i])
	print ""+get_age_path(age[i])+" "+get_gender_path(gender[i])
	train_or_validation="train"
	if(i%4==0):
		train_or_validation="validation"
	src_img=IMDB_PATH+full_path[i][0]
	shutil.copyfile(src_img, AGE_PATH+train_or_validation+"/"+get_age_path(age[i])+"/"+str(i)+".jpg")
	shutil.copyfile(src_img, GENDER_PATH+train_or_validation+"/"+get_gender_path(gender[i])+"/"+str(i)+".jpg")
	shutil.copyfile(src_img, AGE101_PATH+train_or_validation+"/"+format(age[i],"03d")+"/"+str(i)+".jpg")
