

import numpy as np
import medpy.metric._volume as volume
import medpy.metric as metric
import utils
import sys
import os

# force load custom caffe
sys.path.insert(0, '/home/vision/tkdrlf9202/caffe-jonlong/python')
import caffe as caffe
# verify that the model uses custom caffe
print caffe.__file__

# Use GPU for inference: if one wants to use cpu, caffe.set_mode_cpu()
caffe.set_mode_gpu()

# model prototxt and weights location
STEP1_DEPLOY_PROTOTXT = "models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "models/cascadedfcn/step1/step1_weights.caffemodel"
STEP2_DEPLOY_PROTOTXT = "models/cascadedfcn/step2/step2_deploy.prototxt"
STEP2_MODEL_WEIGHTS   = "models/cascadedfcn/step2/step2_weights.caffemodel"

# Load network
net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)

# load dataset
# if full load the data, specify None to num_data_to_load
img_list, lbl_list = utils.load_liver_seg_dataset(data_path='/home/vision/tkdrlf9202/Datasets/liver', num_data_to_load=None)

perf_metrics = []
for idx_subject in range(len(img_list)):
    img = img_list[idx_subject]
    lbl = lbl_list[idx_subject]
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
    v = volume.Volume(pred, lbl_p)
    # calculate metrics as in the oritinal paper
    voe = v.get_volumetric_overlap_error()
    rvd = v.get_relative_volume_difference()
    asd = metric.asd(pred, lbl_p)
    msd = metric.hd(pred, lbl_p)
    dice = metric.dc(pred, lbl_p) * 100 # convert to percentage

    perf_metrics.append([voe, rvd, asd, msd, dice])
    print('subject %d: %s' % (idx_subject, str([voe, rvd, asd, msd, dice])))

perf_metrics = np.array(perf_metrics)
perf_metrics_mean = np.mean(perf_metrics, axis=0)

print('inference complete: mean of performance metrics')
print(perf_metrics_mean)


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

