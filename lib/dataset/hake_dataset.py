#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Oct 22th, 2020            #
#############################################
import torch
import torch.utils
import cv2
import numpy as np
import random
import os
import sys
import inspect
import pickle
import base64
import lmdb
import os.path as osp
import imageio
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# from ult.ult import Augmented_HO_Neg_HICO_pose_pattern_version2, write_hdf5
from ult.ult_Generalized import data_builder, obj2str, str2obj, image_folder_list
from ult.config import cfg

def Generate_relation_bbox(Human, Object):
    ans = [0, 0, 0, 0, 0]
    ans[1] = min(Human[0], Object[0])
    ans[2] = min(Human[1], Object[1])
    ans[3] = max(Human[2], Object[2])
    ans[4] = max(Human[3], Object[3])
    res = np.array(ans).reshape(1,5).astype(np.float64)
    return res

def Generate_action(label, num):
    label = list(label)
    action = np.zeros((1, num), dtype='float64')
    action[:, label] = 1
    return action

def Generate_object(label, num=80):
    object = np.zeros((1, num), dtype='float64')
    if isinstance(label, int) or isinstance(label, np.float64):
        object[:, label-1] = 1
    else:
        idx = np.array(label)
        idx = idx - 1
        object[:, list(idx)] = 1
    return object

def Generate_part_bbox(joint_bbox, Human_bbox=None):
    part_bbox = np.zeros([1, 10, 5], dtype=np.float64)
    if isinstance(joint_bbox, int):
        if Human_bbox is None:
            raise ValueError
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, Human_bbox[0], Human_bbox[1], Human_bbox[2], Human_bbox[3]], dtype=np.float64)
    else:
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, joint_bbox[i]['x1'],  joint_bbox[i]['y1'],  joint_bbox[i]['x2'],  joint_bbox[i]['y2']], dtype=np.float64)
    return part_bbox

class hake_train(torch.utils.data.Dataset):
    def __init__(self, GT_path, Neg_path, Pos_augment=15, Neg_select=60):
        super(hake_train, self).__init__()
        self.GT   = lmdb.open(GT_path)
        self.txn_GT = self.GT.begin(write=False)
        self.Neg  = lmdb.open(Neg_path)
        self.txn_Neg = self.Neg.begin(write=False)
        self.pos_aug = Pos_augment
        self.neg_sel = Neg_select
        self.image_list = [key for key, _ in self.txn_GT.cursor()]
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        current_key_raw  = self.image_list[idx]
        image_id         = current_key_raw.decode()
        current_GT_raw   = self.txn_GT.get(current_key_raw)
        current_GT       = str2obj(current_GT_raw)
    
        im_orig, Human_augmented, Part_bbox, action_PVP0, action_PVP1, action_PVP2, action_PVP3, \
        action_PVP4, action_PVP5, num_pos, gt_10v, action_verb, dataset = data_builder(image_id, current_GT, self.txn_Neg, self.pos_aug, self.neg_sel)

        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        blobs = {}
        blobs['image']   = im_orig
        blobs['H_boxes'] = Human_augmented
        blobs['P_boxes'] = Part_bbox
        Part_bbox[:, :, 1] = np.maximum(Part_bbox[:, :, 1], 0)
        Part_bbox[:, :, 2] = np.maximum(Part_bbox[:, :, 2], 0)
        Part_bbox[:, :, 3] = np.minimum(Part_bbox[:, :, 3], im_shape[1])
        Part_bbox[:, :, 4] = np.minimum(Part_bbox[:, :, 4], im_shape[0])
        blobs['gt_class_P0'] = action_PVP0
        blobs['gt_class_P1'] = action_PVP1
        blobs['gt_class_P2'] = action_PVP2
        blobs['gt_class_P3'] = action_PVP3
        blobs['gt_class_P4'] = action_PVP4
        blobs['gt_class_P5'] = action_PVP5
        blobs['H_num'] = num_pos
        blobs['gt_10v'] = gt_10v
        blobs['gt_verb'] = action_verb
        blobs['dataset'] = dataset
        blobs['image_id'] = image_id
        return blobs

class hake_test_pasta(torch.utils.data.Dataset): # hake-test-full
    def __init__(self, Test_path_hico, Test_path_40v):
        super(hake_test_pasta, self).__init__()
        self.Test_RCNN            = lmdb.open(Test_path_hico)
        self.txn_Test_RCNN        = self.Test_RCNN.begin(write=False)
        self.Test_RCNN_40v        = lmdb.open(Test_path_40v)
        self.txn_Test_RCNN_40v    = self.Test_RCNN_40v.begin(write=False)

        self.hico_image_dir  = image_folder_list['hico-test']
        self.image_list      = [('hico-test', key, osp.join(self.hico_image_dir, 'HICO_test2015_' + (str(int(key.decode()))).zfill(8) + '.jpg')) for key, _ in self.txn_Test_RCNN.cursor()]

        self.v40_image_dir  = image_folder_list['collect']
        self.v40_image_list = [['collect', key, osp.join(self.v40_image_dir, key.decode())] for key, _ in self.txn_Test_RCNN_40v.cursor()]

        self.full_image_list = self.v40_image_list + self.image_list #self.image_list + self.40v_image_list

    def __len__(self):
        return len(self.full_image_list)

    def __getitem__(self, idx):
        dataset_name, current_key_raw, im_path = self.full_image_list[idx]

        im       = self.im_read(im_path)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        test_data = {}
        test_data['image']      = im_orig
        test_data['im_path']    = im_path
        if dataset_name == 'hico-test':
            test_data['test_rcnn']  = str2obj(self.txn_Test_RCNN.get(current_key_raw))
        else:
            test_data['test_rcnn']  = str2obj(self.txn_Test_RCNN_40v.get(current_key_raw))
        test_data['dataset']    = dataset_name
        test_data['im_name']    = current_key_raw.decode()
        return test_data

    def rgba2rgb(self, rgba, background=(255,255,255) ):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros( (row, col, 3), dtype='float32' )
        r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

        a = np.asarray( a, dtype='float32' ) / 255.0

        R, G, B = background

        rgb[:,:,0] = r * a + (1.0 - a) * R
        rgb[:,:,1] = g * a + (1.0 - a) * G
        rgb[:,:,2] = b * a + (1.0 - a) * B

        return np.asarray( rgb, dtype='uint8' )

    def im_read(self, im_path):
        im = cv2.imread(im_path)
        if im is None:
            im = imageio.imread(im_path)
            if im.shape[-1] == 4:
                im = self.rgba2rgb(im)
                im = im[:,:,::-1]
            else:
                raise NotImplementedError
        return im