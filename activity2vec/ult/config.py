#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import copy
import json
from yacs.config import CfgNode as CN

# General configs of directories and model name.
__C = CN()
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.LOG_DIR = osp.join(__C.ROOT_DIR, 'logs')
__C.WEIGHT_DIR = osp.join(__C.ROOT_DIR, 'Weights')
__C.MODEL_NAME = 'default'

# Data configs
__C.DATA = CN()
__C.DATA.NUM_PASTAS = CN()
__C.DATA.NUM_PASTAS.FOOT = 16
__C.DATA.NUM_PASTAS.LEG = 15
__C.DATA.NUM_PASTAS.HIP = 6
__C.DATA.NUM_PASTAS.HAND = 34
__C.DATA.NUM_PASTAS.ARM = 8
__C.DATA.NUM_PASTAS.HEAD = 14
__C.DATA.NUM_PARTS = 10
__C.DATA.NUM_VERBS = 157
__C.DATA.SKELETON_SIZE = 64
__C.DATA.PASTA_NAMES = ['foot', 
                        'leg', 
                        'hip', 
                        'hand', 
                        'arm', 
                        'head']

__C.DATA.FULL_SET_NAMES = ['hico-train', 
                           'hico-test',
                           'hcvrd', 
                           'openimage', 
                           'vcoco', 
                           'pic', 
                           'long_tail_1', 
                           'long_tail_2',
                           'collect']
                           
# Directories of data and weights
__C.DATA.DATA_DIR = osp.join(__C.ROOT_DIR, 'Data')
__C.DATA.ANNO_DB_PATH = osp.join(__C.DATA.DATA_DIR, 'Trainval_HAKE')
__C.DATA.PRED_DB_PATH = osp.join(__C.DATA.DATA_DIR, 'Test_pred_rcnn')
__C.DATA.TEST_GT_PASTA_PATH = osp.join(__C.DATA.DATA_DIR, 'metadata', 'gt_pasta_data.pkl')
__C.DATA.TEST_GT_VERB_PATH = osp.join(__C.DATA.DATA_DIR, 'metadata', 'gt_verb_data.pkl')
__C.DATA.PASTA_LANGUAGE_MATRIX_PATH = osp.join(__C.DATA.DATA_DIR, 'metadata', 'pasta_language_matrix.npy')
__C.DATA.PASTA_WEIGHTS_PATH = osp.join(__C.DATA.DATA_DIR, 'metadata', 'loss_weights.npy')
__C.DATA.IMAGE_FOLDER_LIST = osp.join(__C.DATA.DATA_DIR, 'metadata', 'data_path.json')
__C.DATA.PASTA_NAME_LIST = osp.join(__C.DATA.DATA_DIR, 'metadata', 'Part_State_93_new.txt')
__C.DATA.VERB_NAME_LIST = osp.join(__C.DATA.DATA_DIR, 'metadata', 'verb_list_new.txt')

# Training options
__C.TRAIN = CN()
__C.TRAIN.CHECKPOINT_INTERVAL = 50000
__C.TRAIN.SHOW_INTERVAL = 1000
__C.TRAIN.DISPLAY_INTERVAL = 10
__C.TRAIN.CHECKPOINT_PATH = ''
__C.TRAIN.LOAD_HISTORY = False

# Training params
__C.TRAIN.BASE_LR  = 0.0025
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_SCHEDULE = 'cosine'
__C.TRAIN.MAX_EPOCH = 100
__C.TRAIN.FREEZE_BACKBONE = True
# TODO: multi images per batch(one image for each gpu, this may decrease the performance)
__C.TRAIN.IM_BATCH_SIZE = 1
__C.TRAIN.NUM_WORKERS = 1
__C.TRAIN.HUMAN_PER_IM = 10
__C.TRAIN.POS_RATIO = 0.1
__C.TRAIN.SHOW_LOSS_CURVE = True
__C.TRAIN.WITH_LOSS_WTS = True
__C.TRAIN.LOSS_TYPE = 'bce'
__C.TRAIN.LOSS_WEIGHT_K = 2
__C.TRAIN.COMBINE_PASTA = False
__C.TRAIN.FREEZE_RES4 = True

# Model settings
__C.MODEL = CN()
__C.MODEL.DROPOUT = 0.5
__C.MODEL.MODULE_TRAINED = ['verb']
__C.MODEL.PART_AGG_RULE = [[0, 3], [1, 2], [4], [6, 9], [7, 8], [5]]
__C.MODEL.NUM_FC = 512
__C.MODEL.PART_ROI_ENABLE = True
__C.MODEL.VERB_ONE_MORE_FC = False
__C.MODEL.POSE_MAP = False
__C.MODEL.SKELETON_DIM = 2704

# Benchmark and test settings
__C.BENCHMARK = CN()
__C.BENCHMARK.SHOW_ACTION_RES = False

__C.TEST = CN()
__C.TEST.WEIGHT_PATH = ''
__C.TEST.OUTPUT_DIR = ''
__C.TEST.HUMAN_SCORE_ENHANCE = True
__C.TEST.NUM_WORKERS = 1

# Demo settings
__C.DEMO = CN()
__C.DEMO.YOLO_CFG = osp.join(__C.ROOT_DIR, 'models', 'yolo', 'configs', 'yolov3-spp.cfg')
__C.DEMO.YOLO_WEIGHT = osp.join(__C.ROOT_DIR, 'models', 'yolo', 'checkpoints', 'yolov3-spp.weights')
__C.DEMO.POSE_CFG = osp.join(__C.ROOT_DIR, 'models', 'pose', 'configs', '256x192_res50_lr1e-3_1x.yaml')
__C.DEMO.POSE_WEIGHT = osp.join(__C.ROOT_DIR, 'models', 'pose', 'checkpoints', 'fast_res50_256x192.pth')
__C.DEMO.TRACKER_WEIGHT = osp.join(__C.ROOT_DIR, 'models', 'yolo', 'checkpoints', 'osnet.pth')

# Warning: this config has been deprecated in the newest version.
__C.DEMO.A2V_CFG = osp.join(__C.ROOT_DIR, 'models', 'a2v', 'configs', 'a2v.yaml')

__C.DEMO.A2V_WEIGHT = osp.join(__C.ROOT_DIR, 'models', 'a2v', 'checkpoints', 'pretrained_model.pth')
__C.DEMO.EXCLUDED_VERBS = [57, 146]
__C.DEMO.FONT_PATH = osp.join(__C.ROOT_DIR, 'tools', 'inference_tools', 'consola.ttf')
__C.DEMO.FONT_SIZE = 18
__C.DEMO.MAX_HUMAN_NUM = 4
__C.DEMO.DRAW_SKELETON = False
__C.DEMO.DRAW_RIGID = True
__C.DEMO.DETECTOR = 'yolo'

# Pixel mean values (BGR order) as a (1, 1, 3) array
__C.PIXEL_MEANS = [[[102.9801, 115.9465, 122.7717]]]

# For reproducibility
__C.RNG_SEED = 3

# RoI pooling size
__C.POOLING_SIZE = 7

# GPU id
__C.GPU_ID = 0

# Debug hacker
__C.DEBUG = False

# Config handler.
def get_cfg():
    __C.TRAIN.DATA_SPLITS = [x for x in list(json.load(open(__C.DATA.IMAGE_FOLDER_LIST,'r')).keys()) if 'test' not in x]
    __C.DEMO.EXCLUDED_VERBS = list(set(__C.DEMO.EXCLUDED_VERBS + list(range(117, 157))))
    return __C.clone()