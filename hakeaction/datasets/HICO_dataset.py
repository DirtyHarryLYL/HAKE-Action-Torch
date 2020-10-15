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
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from  .utils.HICO_DET_utils import Augmented_HO_Neg_HICO_pose_pattern_version2, write_hdf5, Get_Next_Instance_HO_Neg_HICO_3D
from   hakeaction.core.utils.pointnet import *
from   prefetch_generator import BackgroundGenerator

class HICO_Trainset(torch.utils.data.Dataset):
    def __init__(self, GT, Neg, cfg, Pos_augment=15, Neg_select=60):
        super(HICO_Trainset, self).__init__()
        self.cfg  = cfg
        self.GT   = GT
        self.Neg  = Neg
        self.pos_aug = Pos_augment
        self.neg_sel = Neg_select
        
    def __len__(self):
        return len(self.GT)

    def __getitem__(self, idx):
        GT       = self.GT[idx]
        image_id = int(GT[0])
        im_file  = self.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)
        
        Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label \
            = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, self.Neg, im_shape, self.pos_aug, self.neg_sel)
            
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

class HICO_Trainset_3D(torch.utils.data.Dataset):
    def __init__(self, GT, Neg, cfg, Pos_augment=15, Neg_select=60):
        super(HICO_Trainset_3D, self).__init__()
        self.cfg     = cfg
        self.GT      = GT
        self.Neg     = Neg
        self.pos_aug = Pos_augment
        self.neg_sel = Neg_select
        self.keys    = list(GT.keys())
        self.att_map = pickle.load(open(self.cfg.DATA_DIR + '/att_map.pkl', 'rb'))
        self.vertex_choice    = \
            np.array(pickle.load(open(cfg.VERTEX_CHOICE_PATH,'rb'), encoding='latin1'))[:,0]
        self.smplx_model_data = \
            pickle.load(open(cfg.SMPLX_MODEL_DATA_PATH, 'rb'), encoding='latin1')
        self.pointNet= PointNetHico().cuda().eval()
        
    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        
    def __getitem__(self, idx):
        image_id = self.keys[idx]
        GT       = self.GT[image_id]
        im_file  = self.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)
        blobs    = Get_Next_Instance_HO_Neg_HICO_3D(im_file, GT, self.Neg, image_id, self.pos_aug, self.neg_sel, \
                                                        self.cfg, self.vertex_choice, self.smplx_model_data, self.pointNet)

        # Move data to device:GPU
        blobs['image_id']    = image_id
        blobs['image']       = torch.from_numpy(im_orig).float().cuda()
        blobs['H_boxes']     = torch.from_numpy(blobs['H_boxes']).float().cuda()
        blobs['O_boxes']     = torch.from_numpy(blobs['O_boxes']).float().cuda()
        blobs['sp']          = torch.from_numpy(blobs['sp']).float().cuda()
        blobs['smplx']       = torch.from_numpy(blobs['smplx']).float().cuda()
        blobs['gt_class_HO'] = torch.from_numpy(blobs['gt_class_HO']).float().cuda()
        blobs['att_2D_map']  = torch.from_numpy(blobs['att_2D_map']).float().cuda()
        blobs['pc']          = torch.from_numpy(blobs['pc']).float().cuda()
        blobs['pc_att_map']  = torch.from_numpy(self.att_map).float().cuda()
        
        return blobs

class HICO_Testset(torch.utils.data.Dataset):
    def __init__(self, testImage, Test_RCNN, cfg, Pos_augment=15, Neg_select=60):
        super(HICO_Testset, self).__init__()
        self.cfg       = cfg
        self.testImage = testImage
        self.Test_RCNN = Test_RCNN 
    
    def __len__(self):
        return len(self.testImage)

    def __getitem__(self, idx):
        line     = self.testImage[idx]
        image_id = int(line[-9:-4])
        im_file  = self.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

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

        im_file  = self.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(im_file)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)

        # print("im_orig.shape:", im_orig.shape)
        Pattern, Human_augmented, _, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, self.Neg, im_shape, self.pos_aug, self.neg_sel)
        
        # tmp
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

        blobs = {}
        blobs['image_id']    = image_id
        blobs['image']       = torch.from_numpy(im_orig).float().cuda()
        blobs['H_boxes']     = torch.from_numpy(Human_augmented).float().cuda()
        blobs['O_boxes']     = torch.from_numpy(Object_augmented).float().cuda()
        blobs['gt_class_HO'] = torch.from_numpy(action_HO).float().cuda()
        blobs['sp']          = torch.from_numpy(Pattern).float().cuda()
        blobs['H_num']       = num_pos
        blobs['binary_label'] = torch.from_numpy(binary_label).float().cuda()

        '''
        print("blobs['O_boxes']: ", blobs['O_boxes'])
        pickle.dump(blobs, open('./tmp_blobs_img1.pkl','wb'))
        write_hdf5('./tmp_blobs_img1.tmp', blobs)
        '''

        return blobs