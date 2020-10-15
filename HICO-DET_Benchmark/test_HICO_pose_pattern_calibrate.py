# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose
from ult.HICO_DET_utils import obj_range

import cv2
import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg):

    # save image information
    This_image = []

    im_orig, _ = get_blob(image_id) 
    blobs = {}

    for Human_out in Test_RCNN[image_id]:
        
        if (Human_out[1] == 'Human'): # This is a valid human
            
            blobs['H_num']   = 0
            blobs['H_boxes'] = [np.empty((0, 5), np.float64)]
            blobs['O_boxes'] = [np.empty((0, 5), np.float64)]
            blobs['sp']      = [np.empty((0, 64, 64, 3), np.float64)]
            blobs['gt_class_O'] = [np.empty((0, 80), np.float64)]

            H_box = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
            index = []

            for i in range(len(Test_RCNN[image_id])):
                Object = Test_RCNN[image_id][i]
                if not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    # 1.the object detection result should > thres  2.the bbox detected is not an object
                    blobs['H_boxes'].append(H_box)
                    blobs['O_boxes'].append(np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5))
                    blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3))
                    classid = Object[4] - 1
                    tmp = np.zeros((1, 80), np.float64)
                    tmp[0, classid] = 1
                    blobs['gt_class_O'].append(tmp)
                    blobs['H_num'] += 1
                    index.append(i)
                    
            blobs['H_boxes'] = np.concatenate(blobs['H_boxes'], axis=0)
            blobs['O_boxes'] = np.concatenate(blobs['O_boxes'], axis=0)
            blobs['sp']      = np.concatenate(blobs['sp'], axis=0)
            blobs['gt_class_O'] = np.concatenate(blobs['gt_class_O'], axis=0)

            if blobs['H_num'] == 0:
                continue

            cls_prob_H, cls_prob_O, cls_prob_sp = net.test_image_separate(sess, im_orig, blobs)
            cls_prob_binary                     = net.test_image_binary(sess, im_orig, blobs)[0]
            
            for i in range(blobs['H_num']):
                Object = Test_RCNN[image_id][index[i]]
                classid = Object[4] - 1
                keys[classid].append(image_id)
                scores_H[classid].append(
                        cls_prob_H[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                scores_O[classid].append(
                        cls_prob_O[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                scores_sp[classid].append(
                        cls_prob_sp[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                hdet[classid].append(np.max(Human_out[5]))
                odet[classid].append(np.max(Object[5]))
                pos[classid].append(cls_prob_binary[i][0])
                neg[classid].append(cls_prob_binary[i][1])
                hbox = np.array(Human_out[2]).reshape(1, -1)
                obox = np.array(Object[2]).reshape(1, -1)
                bboxes[classid].append(np.concatenate([hbox, obox], axis=1))


def test_net(sess, net, Test_RCNN, output_dir):


    np.random.seed(cfg.RNG_SEED)
    keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg = [], [], [], [], [], [], [], [], []
    
    for i in range(80):
        keys.append([])
        scores_H.append([])
        scores_O.append([])
        scores_sp.append([])
        bboxes.append([])
        hdet.append([])
        odet.append([])
    
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])

        im_detect(sess, net, image_id, Test_RCNN, keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1
    
    for i in range(80):
        scores_H[i]  = np.concatenate(scores_H[i], axis=0)
        scores_O[i]  = np.concatenate(scores_O[i], axis=0)
        scores_sp[i] = np.concatenate(scores_sp[i], axis=0)
        bboxes[i] = np.concatenate(bboxes[i], axis=0)
        keys[i]   = np.array(keys[i])
        hdet[i]   = np.array(hdet[i])
        odet[i]   = np.array(odet[i])
        pos[i]    = np.array(pos[i])
        neg[i]    = np.array(neg[i])
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    pickle.dump(scores_H, open(os.path.join(output_dir, 'score_H.pkl'), 'wb'))
    pickle.dump(scores_O, open(os.path.join(output_dir, 'score_O.pkl'), 'wb'))
    pickle.dump(scores_sp, open(os.path.join(output_dir, 'score_sp.pkl'), 'wb'))
    pickle.dump(bboxes, open(os.path.join(output_dir, 'bboxes.pkl'), 'wb'))
    pickle.dump(keys, open(os.path.join(output_dir, 'keys.pkl'), 'wb'))
    pickle.dump(hdet, open(os.path.join(output_dir, 'hdet.pkl'), 'wb'))
    pickle.dump(odet, open(os.path.join(output_dir, 'odet.pkl'), 'wb'))
    pickle.dump(pos, open(os.path.join(output_dir, 'pos.pkl'), 'wb'))
    pickle.dump(neg, open(os.path.join(output_dir, 'neg.pkl'), 'wb'))
    




