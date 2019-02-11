#Merge two dataset with resize

import sys
import os
import shutil

import PIL.Image

if(os.path.exists("./dataset/agegender_appareal/")):
	DATASET_ROOT_PATH="./"
else:
	DATASET_ROOT_PATH="/Volumes/ST5/Keras/"

if len(sys.argv) >= 2:
    DATASET_ROOT_PATH = sys.argv[1]

SRC_FOLDER1=DATASET_ROOT_PATH+"dataset/agegender_vggface2"
SRC_FOLDER2=DATASET_ROOT_PATH+"dataset/agegender_appareal"
DST_FOLDER=DATASET_ROOT_PATH+"dataset/agegender_merged"

IMAGE_SIZE=64

def mergefolders(root_src_dir, root_dst_dir, tag):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        i=0
        for file_ in files:
            if file_==".DS_Store":
                continue
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, tag+"_"+file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            #shutil.copy(src_file, dst_file)

            img = PIL.Image.open(src_file)
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img.save(dst_file)

            print(dst_file+" "+str(i)+"/"+str(len(files)))
            i=i+1

mergefolders(SRC_FOLDER1,DST_FOLDER,"vggface2")
mergefolders(SRC_FOLDER2,DST_FOLDER,"appareal")
