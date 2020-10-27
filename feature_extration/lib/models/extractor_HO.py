from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg

import h5py
import numpy as np
import os
import tqdm
import cv2

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def im_detect(sess, net, image_id, ambi):
    info    = ambi[image_id]
    tot = info['boxes'].shape[0]
    FH_all, FO_all = [np.empty((0, 2048))], [np.empty((0, 2048))]
    if tot == 0:
        return FH_all, FO_all
    filename = info['filename']
    im_file = '/SSD/hico_20160224_det/images/test2015/' + filename
    im      = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig-= cfg.PIXEL_MEANS
    im_orig = im_orig.reshape(1, im_orig.shape[0], im_orig.shape[1], 3)
    object_classes = info['obj_classes']
    holder  = np.zeros((info['boxes'].shape[0], 1))
    boxes   = np.concatenate([holder, info['boxes']], axis=1)
    blobs   = {}
    H_boxes = boxes[np.where(object_classes == 1)[0], :]
    O_boxes = boxes
    blobs['H_boxes'] = H_boxes
    blobs['O_boxes'] = O_boxes
    blobs['H_num']   = tot
    
    FH_all, FO_all = net.get_HO(sess, im_orig, blobs)
    return FH_all, FO_all

def test_net(sess, net, ambi, output_dir):
    
    count = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keys = ambi.keys()
    for key in tqdm.tqdm(ambi.keys()):
        FH, FO = im_detect(sess, net, key, ambi)
        with h5py.File(output_dir + '/' + str(key) + '.h5', 'w') as f:
            f['FH'] = FH
            f['FO'] = FO
