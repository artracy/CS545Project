from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn

sys.setrecursionlimit(4000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


# read the config files for the target and hit models

hit_config_file = 'config_hit.pickle'
target_config_file = 'config_target.pickle'

with open(hit_config_file, 'rb') as f_in:
	hit_C = pickle.load(f_in)

with open(target_config_file, 'rb') as f_in:
	target_C = pickle.load(f_in)


# turn off any data augmentation at test time
hit_C.use_horizontal_flips = False
hit_C.use_vertical_flips = False
hit_C.rot_90 = False

target_C.use_horizontal_flips = False
target_C.use_vertical_flips = False
target_C.rot_90 = False

img_path = options.test_path




def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


# Start of getting the data
num_features = 1024

############################
# Start of target data
############################

target_class_mapping = target_C.class_mapping

if 'bg' not in target_class_mapping:
	target_class_mapping['bg'] = len(target_class_mapping)

target_class_mapping = {v: k for k, v in target_class_mapping.items()}
print(target_class_mapping)
target_class_to_color = {target_class_mapping[v]: np.random.randint(0, 255, 3) for v in target_class_mapping}
target_C.num_rois = int(options.num_rois)


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


target_img_input = Input(shape=input_shape_img)
target_roi_input = Input(shape=(target_C.num_rois, 4))
target_feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
target_shared_layers = nn.nn_base(target_img_input, trainable=True)

# define the RPN, built on the base layers
target_num_anchors = len(target_C.anchor_box_scales) * len(target_C.anchor_box_ratios)
target_rpn_layers = nn.rpn(target_shared_layers, target_num_anchors)

target_classifier = nn.classifier(target_feature_map_input, target_roi_input, target_C.num_rois, nb_classes=len(target_class_mapping), trainable=True)

target_model_rpn = Model(target_img_input, target_rpn_layers)
target_model_classifier_only = Model([target_feature_map_input, target_roi_input], target_classifier)

target_model_classifier = Model([target_feature_map_input, target_roi_input], target_classifier)

print('Loading weights from {}'.format(target_C.model_path))
target_model_rpn.load_weights(target_C.model_path, by_name=True)
target_model_classifier.load_weights(target_C.model_path, by_name=True)

target_model_rpn.compile(optimizer='sgd', loss='mse')
target_model_classifier.compile(optimizer='sgd', loss='mse')

#######################################
#  Start of the Hits model and network
#######################################

hit_class_mapping = hit_C.class_mapping

if 'bg' not in hit_class_mapping:
	hit_class_mapping['bg'] = len(hit_class_mapping)

hit_class_mapping = {v: k for k, v in hit_class_mapping.items()}
print(hit_class_mapping)
hit_class_to_color = {hit_class_mapping[v]: np.random.randint(0, 255, 3) for v in hit_class_mapping}
hit_C.num_rois = int(options.num_rois)


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


hit_img_input = Input(shape=input_shape_img)
hit_roi_input = Input(shape=(hit_C.num_rois, 4))
hit_feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
hit_shared_layers = nn.nn_base(hit_img_input, trainable=True)

# define the RPN, built on the base layers
hit_num_anchors = len(hit_C.anchor_box_scales) * len(hit_C.anchor_box_ratios)
hit_rpn_layers = nn.rpn(hit_shared_layers, hit_num_anchors)

hit_classifier = nn.classifier(hit_feature_map_input, hit_roi_input, hit_C.num_rois, nb_classes=len(hit_class_mapping), trainable=True)

hit_model_rpn = Model(hit_img_input, hit_rpn_layers)
hit_model_classifier_only = Model([hit_feature_map_input, hit_roi_input], hit_classifier)

hit_model_classifier = Model([hit_feature_map_input, hit_roi_input], hit_classifier)

print('Loading weights from {}'.format(hit_C.model_path))
hit_model_rpn.load_weights(hit_C.model_path, by_name=True)
hit_model_classifier.load_weights(hit_C.model_path, by_name=True)

hit_model_rpn.compile(optimizer='sgd', loss='mse')
hit_model_classifier.compile(optimizer='sgd', loss='mse')



# Clear Data for results

all_imgs = []

target_classes = {}
hit_classes = {}

bbox_threshold = 0.8

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    #Parse the next file
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)
    
    #load in the image to process
	img = cv2.imread(filepath)
    
	X, ratio = format_img(img, hit_C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))
        
        
    ########################
    # Collect Hit Data
    ########################

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = hit_model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, hit_C, K.image_dim_ordering(),max_boxes=100, overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	hit_bboxes = {}
	hit_probs = {}

	for jk in range(R.shape[0]//hit_C.num_rois + 1):
		ROIs = np.expand_dims(R[hit_C.num_rois*jk:hit_C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//hit_C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],hit_C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = hit_model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = hit_class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in hit_bboxes:
				hit_bboxes[cls_name] = []
				hit_probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= hit_C.classifier_regr_std[0]
				ty /= hit_C.classifier_regr_std[1]
				tw /= hit_C.classifier_regr_std[2]
				th /= hit_C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			hit_bboxes[cls_name].append([hit_C.rpn_stride*x, hit_C.rpn_stride*y, hit_C.rpn_stride*(x+w), hit_C.rpn_stride*(y+h)])
			hit_probs[cls_name].append(np.max(P_cls[0, ii, :]))

    ########################
    # Collect Target Data
    ########################

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = target_model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, target_C, K.image_dim_ordering(),max_boxes=100, overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	target_bboxes = {}
	target_probs = {}

	for jk in range(R.shape[0]//target_C.num_rois + 1):
		ROIs = np.expand_dims(R[target_C.num_rois*jk:target_C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//target_C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],target_C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = target_model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = target_class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in target_bboxes:
				target_bboxes[cls_name] = []
				target_probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= target_C.classifier_regr_std[0]
				ty /= target_C.classifier_regr_std[1]
				tw /= target_C.classifier_regr_std[2]
				th /= target_C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			target_bboxes[cls_name].append([target_C.rpn_stride*x, target_C.rpn_stride*y, target_C.rpn_stride*(x+w), target_C.rpn_stride*(y+h)])
			target_probs[cls_name].append(np.max(P_cls[0, ii, :]))

    #############################################
    #  Output the new image and draw pictures
    #############################################


	all_dets = []
    
    #############################################
    #  Hit boxes
    #############################################

	for key in hit_bboxes:
		bbox = np.array(hit_bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(hit_probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2) 

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(hit_class_to_color[key][0]), int(hit_class_to_color[key][1]), int(hit_class_to_color[key][2])),20)

			textLabel = '{}'.format(key)
			

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
			
            
    #############################################
    #  Target boxes
    #############################################

	for key in target_bboxes:
		bbox = np.array(target_bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(target_probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2) 

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(target_class_to_color[key][0]), int(target_class_to_color[key][1]), int(target_class_to_color[key][2])),20)

			textLabel = '{}'.format(key)			

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
			

	print('Elapsed time = {}'.format(time.time() - st))

	#cv2.imshow('img', img)
	#cv2.waitKey(0)
	cv2.imwrite('result\{}_result.png'.format(img_name),img)
