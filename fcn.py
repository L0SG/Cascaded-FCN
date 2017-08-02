# model prototxt and weights location
STEP1_DEPLOY_PROTOTXT = "models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "models/cascadedfcn/step1/step1_weights.caffemodel"
STEP2_DEPLOY_PROTOTXT = "models/cascadedfcn/step2/step2_deploy.prototxt"
STEP2_MODEL_WEIGHTS   = "models/cascadedfcn/step2/step2_weights.caffemodel"

import numpy as np
import medpy.metric._volume as volume
import medpy.metric as metric
import utils
import sys
sys.path.append('/home/tkdrlf9202/caffe-jonlong/python')
import caffe
# verify that the model uses custom caffe
print caffe.__file__

# Use GPU for inference: if one wants to use cpu, caffe.set_mode_cpu()
caffe.set_mode_gpu()

# Load network
net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)

img_list, lbl_list = utils.load_liver_seg_dataset(data_path='/home/tkdrlf9202/Datasets/liver')

"""
# load the dicom and mask
img = utils.read_dicom_series("/home/tkdrlf9202/Datasets/liver_segmentation/dicom_sample/", filepattern="P_*")
lbl = utils.read_liver_seg_masks_raw("/home/tkdrlf9202/Datasets/liver_segmentation/seg_sample/", img.shape)
# since our data has range of 0~2048 (instead of -1024~1024) subtract 1024
img = np.add(img, -1024)

# load routine for ref. dataset
img = utils.read_dicom_series("/home/tkdrlf9202/PycharmProjects/Cascaded-FCN/test_image/3Dircadb1.17/PATIENT_DICOM", filepattern="image_*")
lbl = utils.read_liver_lesion_masks("/home/tkdrlf9202/PycharmProjects/Cascaded-FCN/test_image/3Dircadb1.17/MASKS_DICOM")
"""

perf_metrics = []
for idx in range(len(img_list)):
    img = img_list[idx]
    lbl = lbl_list[idx]
    # since our data has range of 0~2048 (instead of -1024~1024) subtract 1024
    img = np.add(img, -1024)

    # preproess the image slice
    img_p = np.zeros((572, 572, img.shape[2]), dtype=np.float)
    lbl_p = np.zeros((388, 388, lbl.shape[2]), dtype=np.uint8)
    for idx in xrange(img.shape[2]):
        img_p[..., idx] = utils.step1_preprocess_img_slice(img[..., idx])
        lbl_p[..., idx] = utils.preprocess_lbl_slice(lbl[..., idx])

    # predict
    pred = []
    for idx in xrange(img.shape[2]):
        net1.blobs['data'].data[0,0,...] = img_p[..., idx]
        pred.append(net1.forward()['prob'][0, 1] > 0.5)

    pred = np.array(pred).transpose(1, 2, 0)

    # create volume instance from medpy
    v = volume(lbl_p, pred)
    # calculate metrics
    voe = v.get_volumetric_overlap_error()
    rvd = v.get_relative_volume_difference()
    asd = metric.asd(lbl_p, pred)
    msd = metric.hd(lbl_p, pred)
    dice = metric.dc(lbl_p, pred)

    perf_metrics.append([voe, rvd, asd, msd, dice])



"""
# visualize the results
for idx in xrange(30, 100, 20):
    utils.imshow(img[...,idx], img_p[..., idx], lbl_p[...,idx], pred[...,idx])
"""
# delete the network
del net1

"""
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
"""

