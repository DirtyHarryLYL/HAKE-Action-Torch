from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
global_cfg = __C

'''
    Global Dir settings
'''
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.WEIGHT_DIR = __C.ROOT_DIR + "/Weight"
__C.LOG_DIR    = __C.ROOT_DIR + "/logs"
for DIR in [__C.WEIGHT_DIR, __C.LOG_DIR]:
    if not os.path.exists(DIR):
        os.mkdir(DIR)