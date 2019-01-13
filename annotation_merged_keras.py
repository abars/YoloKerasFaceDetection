#Merge two dataset

import sys
import os
import shutil

if(os.path.exists("./dataset/agegender_appareal/")):
	DATASET_ROOT_PATH="./"
else:
	DATASET_ROOT_PATH="/Volumes/ST5/Keras/"

SRC_FOLDER1=DATASET_ROOT_PATH+"dataset/agegender_vggface2"
SRC_FOLDER2=DATASET_ROOT_PATH+"dataset/agegender_appareal"
DST_FOLDER=DATASET_ROOT_PATH+"dataset/agegender_merged"

def mergefolders(root_src_dir, root_dst_dir, tag):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            print(tag+"_"+file_)
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, tag+"_"+file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)

mergefolders(SRC_FOLDER1,DST_FOLDER,"vggface2")
mergefolders(SRC_FOLDER2,DST_FOLDER,"appareal")
