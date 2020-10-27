from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer

import h5py
import numpy as np
import os
import tqdm
import cv2

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def im_detect(sess, net, image_id, ambi):
    info    = ambi[image_id]
    FR_all = [np.empty((0, 4096))]
    if info['pair_ids'].shape[0] <= 0:
        return FR_all
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
    tot = info['pair_ids'].shape[0]
    H_boxes = boxes[list(info['pair_ids'][:, 0])]
    O_boxes = boxes[list(info['pair_ids'][:, 1])]
    R_boxes = np.zeros((tot, 5))
    R_boxes[:, 1] = np.minimum(H_boxes[:, 1], O_boxes[:, 1])
    R_boxes[:, 2] = np.minimum(H_boxes[:, 2], O_boxes[:, 2])
    R_boxes[:, 3] = np.maximum(H_boxes[:, 3], O_boxes[:, 3])
    R_boxes[:, 4] = np.maximum(H_boxes[:, 4], O_boxes[:, 4])
    
    l = 0
    r = min(tot, l + 100)
    while l < tot:
        blobs['H_num'] = r - l
        blobs['R_boxes'] = R_boxes[l:r, :]
        FR = net.get_relation(sess, im_orig, blobs)[0]
        FR_all.append(FR)
        l = r
        r = min(tot, l + 100)
    FR_all = np.concatenate(FR_all, axis=0)
    return FR_all
    
def test_net(sess, net, ambi, output_dir):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keys = ambi.keys()
    for key in tqdm.tqdm(ambi.keys()):
        FR = im_detect(sess, net, key, ambi)
        with h5py.File(output_dir + '/' + str(key) + '.h5', 'w') as f:
            f['R'] = FR

