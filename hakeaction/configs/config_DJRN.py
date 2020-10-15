from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from .config_global import global_cfg

__C = edict()
cfg = __C

__C.TRAIN = edict()
__C.TRAIN_MODULE = 1
__C.TRAIN_MODULE_UPDATE = 1
__C.TRAIN_INIT_WEIGHT = 3
__C.TRAIN_MODULE_CONTINUE = 2
__C.TRAIN.LEARNING_RATE = 0.0001 # 0.001 # 0.05 0.008 0.001 0.0001
__C.TRAIN_DROP_OUT_BINARY = 0.8
__C.TRAIN.SNAPSHOT_ITERS = 100000
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.GAMMA = 0.96
__C.TRAIN.STEPSIZE = 20000
__C.TRAIN.SNAPSHOT_KEPT = None
__C.TRAIN.DISPLAY = 10
__C.TRAIN.SUMMARY_INTERVAL = 200
__C.TRAIN.MAX_EPOCH = 1000
__C.TRAIN.BATCHSIZE = 1

__C.RESNET = edict()
__C.RESNET.MAX_POOL = False
__C.RESNET.FIXED_BLOCKS = 1
__C.LR_DECAY = edict()
__C.LR_DECAY.TYPE = 'none'
__C.LR_DECAY.STEPS = 5.0
__C.LR_DECAY.T_MUL = 2.0
__C.LR_DECAY.M_MUL = 1.0
__C.LANG_NOISE = 0
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.RNG_SEED = 3

__C.ROOT_DIR   = global_cfg.ROOT_DIR
# __C.DATA_DIR = "/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data"
# __C.DATA_DIR = "/Disk1/yonglu/TIN_CVPR_final/Data"

__C.DATA_DIR   = "/Disk1/yonglu/DJR/Data"
# __C.WEIGHT_DIR = __C.ROOT_DIR + "/Weight/DJRN/"
__C.WEIGHT_DIR = "/Disk4/DJRN_weight_tmp/"
# __C.LOG_DIR    = __C.ROOT_DIR + "/logs/DJRN/"
__C.LOG_DIR    = "/Disk4/DJRN_weight_tmp/"
__C.SMPLX_PATH = "/Disk3/zhanke/data/smplx_train"
__C.SMPLX_TEST_PATH       = "/Disk3/zhanke/data/smplx_test"
__C.SMPLX_MODEL_DATA_PATH = "/Disk1/yonglu/SMPL-X/smplx/smplx/SMPLX_MALE.pkl"
__C.VERTEX_CHOICE_PATH    = './hakeaction/datasets/utils/hico_files/remaining_vertexs-part.pkl'
for DIR in [__C.WEIGHT_DIR, __C.LOG_DIR]:
    if not os.path.exists(DIR):
        os.mkdir(DIR)

__C.EXP_DIR       = 'default'
__C.USE_GPU_NMS   = True
__C.POOLING_MODE  = 'crop'
__C.POOLING_SIZE  = 7
__C.ANCHOR_SCALES = [8,16,32]
__C.ANCHOR_RATIOS = [0.5,1,2] 
__C.RPN_CHANNELS  = 512
__C.ONLY_KEEP_BEST_WEIGHT = False  # If True, training script will only keep the best ckpt and delete others. 
__C.PARTS_TO_FREEZE =  ['image_to_head',
                        'resnet_layer4',
                        'resnet_layer5',
                        'head_phi',
                        'head_g',
                        'attention_pool_layer_H_network',
                        'attention_pool_layer_O_network',
                        'head_bottleneck',
                        'fc8_SH',
                        'fc9_SH',
                        'fc8_SO',
                        'fc9_SO',
                        'cls_score_H',
                        'cls_score_O',
                        'cls_score_sp',
                        ]
# dataloader settings
__C.DATA_REQUIRE_PC    = False
__C.DATA_REQUIRE_SMPLX = True

# loss settings
__C.TRAIN_2D_BRANCH = False
__C.TRAIN_3D_BRANCH = False
__C.TRAIN_BSH_BRANCH = True
__C.BRANCH_2D_LOSS_TERMS = ['sp', 'H', 'O'] # avaliable: 'sp', 'H', 'O'
__C.ATTENTION_LOSS = 'KL'                   # avaliable: 'KL', 'MSE'
assert(__C.TRAIN_2D_BRANCH or __C.TRAIN_3D_BRANCH or __C.TRAIN_BSH_BRANCH)