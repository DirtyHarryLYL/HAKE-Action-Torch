import torch
import torch.utils
import cv2
import numpy as np
import random
import os
import sys
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ult.ult import Augmented_HO_Neg_HICO_pose_pattern_version2, write_hdf5
from ult.config_TIN import cfg
# from ult.u

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
    object = np.zeros((1, num), dypte='flat64')
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

class HICO_Trainset(torch.utils.data.Dataset):
    def __init__(self, GT, Neg, Pos_augment=15, Neg_select=60):
        super(HICO_Trainset, self).__init__()
        self.GT   = GT
        self.Neg  = Neg
        self.pos_aug = Pos_augment
        self.neg_sel = Neg_select
    
    def __len__(self):
        return len(self.GT)

    def __getitem__(self, idx):
        GT       = self.GT[idx]

        image_id = int(GT[0])
        # image_id = 1

        im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, self.Neg, im_shape, self.pos_aug, self.neg_sel)
        
        blobs = {}
        blobs['image_id']    = image_id
        blobs['image']       = torch.from_numpy(im_orig).float().cuda()
        blobs['H_boxes']     = torch.from_numpy(Human_augmented).float().cuda()
        blobs['O_boxes']     = torch.from_numpy(Object_augmented).float().cuda()
        blobs['gt_class_HO'] = torch.from_numpy(action_HO).float().cuda()
        blobs['sp']          = torch.from_numpy(Pattern).float().cuda()
        blobs['H_num']       = num_pos
        blobs['binary_label'] = torch.from_numpy(binary_label).float().cuda()


        # TODO: Move data to GPU
        return blobs

class HICO_Testset(torch.utils.data.Dataset):
    def __init__(self, testImage, Test_RCNN, Pos_augment=15, Neg_select=60):
        super(HICO_Testset, self).__init__()

        self.testImage = testImage
        self.Test_RCNN = Test_RCNN 
    
    def __len__(self):
        return len(self.testImage)

    def __getitem__(self, idx):
        line     = self.testImage[idx]

        image_id = int(line[-9:-4])
        im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        # blobs = {}
        # blobs['H_num'] = 1

        test_data = {}
        test_data['image']      = torch.from_numpy(im_orig).float().cuda()
        test_data['image_id']   = image_id

        return test_data

class HICO_DEBUG(torch.utils.data.Dataset):
    def __init__(self, GT, Neg, Pos_augment=15, Neg_select=60):
        super(HICO_DEBUG, self).__init__()
        self.GT   = GT
        self.Neg  = Neg
        self.pos_aug = Pos_augment
        self.neg_sel = Neg_select
    
    def __len__(self):
        return len(self.GT)

    def __getitem__(self, idx):
        idx = 0
        GT  = self.GT
        image_id = int(GT[0])

        assert(image_id == 1)

        im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        # print("im_orig.shape:", im_orig.shape)

        Pattern, Human_augmented, _, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, self.Neg, im_shape, self.pos_aug, self.neg_sel)
        
        # tmp
        # '''
        Object_augmented = np.array([[0, 58.      ,  97.      , 571.      , 404.      ],
                                    [0,  58.      ,  98.      , 578.      , 394.      ],
                                    [0,  60.      ,  99.      , 570.      , 400.      ],
                                    [0,  76.      , 114.      , 582.      , 395.      ],
                                    [0, 205.      ,  32.      , 426.      , 305.      ],
                                    [0, 207.      ,  32.      , 426.      , 299.      ],
                                    [0, 208.      ,  25.      , 443.      , 316.      ],
                                    [0, 212.      ,  19.      , 437.      , 356.      ],
                                    [0, 216.764816,  35.14827 , 442.983093, 318.884949],
                                    [0,  47.103058,  90.991562, 580.514649, 403.134949]])

        # '''

        blobs = {}
        blobs['image_id']    = image_id
        blobs['image']       = torch.from_numpy(im_orig).float().cuda()
        blobs['H_boxes']     = torch.from_numpy(Human_augmented).float().cuda()
        blobs['O_boxes']     = torch.from_numpy(Object_augmented).float().cuda()
        blobs['gt_class_HO'] = torch.from_numpy(action_HO).float().cuda()
        blobs['sp']          = torch.from_numpy(Pattern).float().cuda()
        blobs['H_num']       = num_pos
        blobs['binary_label'] = torch.from_numpy(binary_label).float().cuda()

        # print("blobs['O_boxes']: ", blobs['O_boxes'])
        # pickle.dump(blobs, open('./tmp_blobs_img1.pkl','wb'))
        # write_hdf5('./tmp_blobs_img1.tmp', blobs)


        # TODO: Move data to GPU
        return blobs