# ----------------------------------------------
# Reference gender from camera face
# (Quote from https://github.com/xingwangsfu/caffe-yolo)
# ----------------------------------------------

import caffe
from datetime import datetime
import numpy as np
import sys, getopt
import cv2
import os

#os.environ['KERAS_BACKEND'] = 'tensorflow'

import plaidml.keras
plaidml.keras.install_backend()

from keras.models import load_model
from keras.preprocessing import image

def interpret_output(output, img_width, img_height):
	classes = ["face"]
	w_img = img_width
	h_img = img_height
	threshold = 0.2
	iou_threshold = 0.5
	num_class = 1
	num_box = 2
	grid_size = 11
	probs = np.zeros((grid_size,grid_size,2,20))
	class_probs = np.reshape(output[0:grid_size*grid_size*num_class],(grid_size,grid_size,num_class))
	scales = np.reshape(output[grid_size*grid_size*num_class:grid_size*grid_size*num_class+grid_size*grid_size*2],(grid_size,grid_size,2))
	boxes = np.reshape(output[grid_size*grid_size*num_class+grid_size*grid_size*2:],(grid_size,grid_size,2,4))
	offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*grid_size*2),(2,grid_size,grid_size)),(1,2,0))

	boxes[:,:,:,0] += offset
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / grid_size
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
		
	boxes[:,:,:,0] *= w_img
	boxes[:,:,:,1] *= h_img
	boxes[:,:,:,2] *= w_img
	boxes[:,:,:,3] *= h_img

	for i in range(2):
		for j in range(num_class):
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
	filter_mat_probs = np.array(probs>=threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	probs_filtered = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

	argsort = np.array(np.argsort(probs_filtered))[::-1]
	boxes_filtered = boxes_filtered[argsort]
	probs_filtered = probs_filtered[argsort]
	classes_num_filtered = classes_num_filtered[argsort]
		
	for i in range(len(boxes_filtered)):
		if probs_filtered[i] == 0 : continue
		for j in range(i+1,len(boxes_filtered)):
			if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
				probs_filtered[j] = 0.0
		
	filter_iou = np.array(probs_filtered>0.0,dtype='bool')
	boxes_filtered = boxes_filtered[filter_iou]
	probs_filtered = probs_filtered[filter_iou]
	classes_num_filtered = classes_num_filtered[filter_iou]

	result = []
	for i in range(len(boxes_filtered)):
		result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

	return result

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def show_results(MODE,img,results, img_width, img_height, net_age, net_gender, net_emotion, model_age, model_gender, model_emotion):
	img_cp = img.copy()
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2

		if(w<h):
			w=h
		else:
			h=w

		xmin = x-w
		xmax = x+w
		ymin = y-h
		ymax = y+h
		if xmin<0:
			xmin = 0
		if ymin<0:
			ymin = 0
		if xmax>img_width:
			xmax = img_width
		if ymax>img_height:
			ymax = img_height

		cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
		cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
		cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)	

		target_image=img_cp
		margin=w/4

		x=xmin
		y=ymin
		w*=2
		h*=2

		x2=x-margin
		y2=y-margin
		w2=w+margin*2
		h2=h+margin*2

		if(x2<0):
			x2=0
		if(y2<0):
			y2=0
		if(x2+w2>=target_image.shape[1]):
			w2=target_image.shape[1]-1-x2
		if(y2+h2>=target_image.shape[0]):
			h2=target_image.shape[0]-1-y2

		face_image = target_image[y2:y2+h2, x2:x2+w2]

		if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
			continue

		IMAGE_SIZE=227
		IMAGE_SIZE_KERAS=224
		IMAGE_SIZE_FER2013=64

		img = cv2.resize(face_image, (IMAGE_SIZE,IMAGE_SIZE))
		img = np.expand_dims(img, axis=0)
		img = img - (104,117,123) #BGR mean value of VGG16
		img = img.transpose((0, 3, 1, 2))

		img_fer2013 = cv2.resize(face_image, (IMAGE_SIZE_FER2013,IMAGE_SIZE_FER2013))
		img_fer2013 = cv2.cvtColor(img_fer2013,cv2.COLOR_BGR2GRAY)
		img_fer2013 = np.expand_dims(img_fer2013, axis=0)
		img_fer2013 = np.expand_dims(img_fer2013, axis=3)
		img_fer2013 = img_fer2013 / 255.0 *2 -1

		img_keras = cv2.resize(face_image, (IMAGE_SIZE_KERAS,IMAGE_SIZE_KERAS))
		img_keras = img_keras[::-1, :, ::-1].copy()	#BGR to RGB
		img_keras = np.expand_dims(img_keras, axis=0)
		img_keras = img_keras / 255.0

		caffe_final_layer="prob"
		gender_revert=True
		if(MODE=="converted"):
			caffe_final_layer="dense_2"
			img = img_keras.copy()
			img = img.transpose((0, 3, 1, 2))
			gender_revert = False

		cv2.rectangle(target_image, (x2,y2), (x2+w2,y2+h2), color=(0,0,255), thickness=3)
		offset=16

		lines_age=open('words/agegender_age_words.txt').readlines()
		lines_gender=open('words/agegender_gender_words.txt').readlines()
		lines_fer2013=open('words/emotion_words.txt').readlines()

		if(net_age!=None):
			out = net_age.forward_all(data = img)
			pred_age = out[caffe_final_layer]
			prob_age = np.max(pred_age)
			cls_age = pred_age.argmax()
			cv2.putText(target_image, "Caffe : %.2f" % prob_age + " " + lines_age[cls_age], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(net_gender!=None):
			out = net_gender.forward_all(data = img)
			pred_gender = out[caffe_final_layer]
			prob_gender = np.max(pred_gender)
			if(gender_revert):
				cls_gender = 1-pred_gender.argmax()
			else:
				cls_gender = pred_gender.argmax()
			cv2.putText(target_image, "Caffe : %.2f" % prob_gender + " " + lines_gender[cls_gender], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(net_emotion!=None):
			out = net_emotion.forward_all(data = img_fer2013.transpose((0, 3, 1, 2)))
			pred_emotion = out["global_average_pooling2d_1"]
			prob_emotion = np.max(pred_emotion)
			cls_emotion = pred_emotion.argmax()
			cv2.putText(target_image, "Caffe : %.2f" % prob_emotion + " " + lines_fer2013[cls_emotion], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(model_age!=None):
			pred_age_keras = model_age.predict(img_keras)[0]
			prob_age_keras = np.max(pred_age_keras)
			cls_age_keras = pred_age_keras.argmax()
			cv2.putText(target_image, "Keras : %.2f" % prob_age_keras + " " + lines_age[cls_age_keras], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(model_gender!=None):
			pred_gender_keras = model_gender.predict(img_keras)[0]
			prob_gender_keras = np.max(pred_gender_keras)
			cls_gender_keras = pred_gender_keras.argmax()
			cv2.putText(target_image, "Keras : %.2f" % prob_gender_keras + " " + lines_gender[cls_gender_keras], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(model_emotion!=None):
			pred_emotion_keras = model_emotion.predict(img_fer2013)[0]
			prob_emotion_keras = np.max(pred_emotion_keras)
			cls_emotion_keras = pred_emotion_keras.argmax()
			cv2.putText(target_image, "Keras : %.2f" % prob_emotion_keras + " " + lines_fer2013[cls_emotion_keras], (x2,y2+h2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

	cv2.imshow('YOLO detection',img_cp)
	
	#if(DEMO_IMG!=""):
	#	cv2.imwrite("detection.jpg", img_cp)
	#	cv2.waitKey(1000)

def get_mean(binary_proto,width,height):
	mean_filename=binary_proto
	proto_data = open(mean_filename, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]
	print "mean value of "+binary_proto+" is "+str(mean)+" shape "+str(mean.shape)

	shape=(mean.shape[0],height,width);
	mean=mean.copy()
	mean.resize(shape)

	print "resized mean value is "+str(mean)

	return mean

def main(argv):
	MODE="caffe"
	DEMO_IMG=""

	if len(sys.argv) >= 2:
		MODE = sys.argv[1]
		if(len(sys.argv)>=3):
			DEMO_IMG=sys.argv[2]
	else:
		print("usage: python agegender_demo.py [caffe/keras/caffekeras/converted]")
		sys.exit(1)
	if(MODE!="caffe" and MODE!="keras" and MODE!="caffekeras" and MODE!="converted"):
		print("Unknown mode "+MODE)
		sys.exit(1)

	model_filename = './pretrain/face.prototxt'
	weight_filename = './pretrain/face.caffemodel'

	net = caffe.Net(model_filename, weight_filename, caffe.TEST)

	if(MODE == "caffe" or MODE == "caffekeras"):
		net_age  = caffe.Net('./pretrain/deploy_age.prototxt', './pretrain/age_net.caffemodel', caffe.TEST)
		net_gender  = caffe.Net('./pretrain/deploy_gender.prototxt', './pretrain/gender_net.caffemodel', caffe.TEST)
		net_emotion = caffe.Net('./pretrain/emotion_miniXception.prototxt', './pretrain/emotion_miniXception.caffemodel', caffe.TEST)
	else:
		net_age=None
		net_gender=None
		net_emotion=None

	if(MODE == "converted"):
		net_age  = caffe.Net('./pretrain/agegender_age_vgg16.prototxt', './pretrain/agegender_age_vgg16.caffemodel', caffe.TEST)
		net_gender  = caffe.Net('./pretrain/agegender_gender_vgg16.prototxt', './pretrain/agegender_gender_vgg16.caffemodel', caffe.TEST)

	if(MODE == "keras" or MODE == "caffekeras"):
		model_age = load_model('./pretrain/train_age_vgg16.hdf5')
		model_gender = load_model('./pretrain/train_gender_vgg16.hdf5')
		#if(os.path.exists('./pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5')):
		#	model_emotion = load_model('./pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5')
		#else:
		model_emotion = None
	else:
		model_age = None
		model_gender = None
		model_emotion = None

	while True:
		cap = cv2.VideoCapture(0)
		ret, frame = cap.read() #BGR
		img=frame
		img = img[...,::-1]  #BGR 2 RGB
		inputs = img.copy() / 255.0
		
		if(DEMO_IMG!=""):
			img = caffe.io.load_image(DEMO_IMG) # load the image using caffe io
			inputs = img
		
		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
		img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		results = interpret_output(out['layer20-fc'][0], img.shape[1], img.shape[0])
		show_results(MODE,img_cv,results, img.shape[1], img.shape[0], net_age, net_gender, net_emotion, model_age, model_gender, model_emotion)

		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	main(sys.argv[1:])
