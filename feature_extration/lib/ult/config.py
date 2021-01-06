from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.RESNET = edict()

__C.RESNET.MAX_POOL = False

__C.RESNET.FIXED_BLOCKS = 1

__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

__C.RNG_SEED = 3

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'Data'))

__C.POOLING_MODE = 'crop'

__C.POOLING_SIZE = 7

__C.ANCHOR_SCALES = [8,16,32]

__C.ANCHOR_RATIOS = [0.5,1,2]

__C.RPN_CHANNELS = 512

__C.TRAIN = edict()

__C.TRAIN.WEIGHT_DECAY = 1e-2
