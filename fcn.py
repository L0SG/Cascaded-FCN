# model prototxt and weights location
STEP1_DEPLOY_PROTOTXT = "models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "models/cascadedfcn/step1/step1_weights.caffemodel"
STEP2_DEPLOY_PROTOTXT = "models/cascadedfcn/step2/step2_deploy.prototxt"
STEP2_MODEL_WEIGHTS   = "models/cascadedfcn/step2/step2_weights.caffemodel"

import sys
sys.path.append('/home/tkdrlf9202/caffe-jonlong/python')
import caffe
print caffe.__file__
# Use CPU for inference
# caffe.set_mode_cpu()
# Use GPU for inference
caffe.set_mode_gpu()

import numpy as np
import utils

# load the dicom and mask
img = utils.read_dicom_series("/home/tkdrlf9202/Datasets/liver_segmentation/dicom_sample/")
lbl = utils.read_liver_seg_masks_raw("/home/tkdrlf9202/Datasets/liver_segmentation/seg_sample/", img.shape)
# lbl = utils.read_liver_lesion_masks("/home/tkdrlf9202/Datasets/liver_segmentation/seg_sample/")
print(img.shape, lbl.shape)

# preproess the image slice
S = 90
img_p = utils.step1_preprocess_img_slice(img[...,S])
lbl_p = utils.preprocess_lbl_slice(lbl[...,S])
for s in range(50,100,20):
    utils.imshow(img[...,s],lbl[...,s])

# Load network
net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)

# predict
net1.blobs['data'].data[0,0,...] = img_p
pred = net1.forward()['prob'][0,1] > 0.5

# delete the network
del net1

# preprocess the liver patch for step2
img_p2, bbox = utils.step2_preprocess_img_slice(img_p, pred)

# Load step2 network
net2 = caffe.Net(STEP2_DEPLOY_PROTOTXT, STEP2_MODEL_WEIGHTS, caffe.TEST)

# Predict
net2.blobs['data'].data[0,0,...] = img_p2
pred2 = net2.forward()['prob'][0,1]

# Visualize result

# extract liver portion as predicted by net1
x1,x2,y1,y2 = bbox
lbl_p_liver = lbl_p[y1:y2,x1:x2]
# Set labels to 0 and 1
lbl_p_liver[lbl_p_liver == 1] = 0
lbl_p_liver[lbl_p_liver == 2] = 1

del net2