from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import logging
import time
import glob
from tqdm import tqdm
import os
import sys
import inspect
import deepdish as dd
import copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from lib.networks.pasta_net import pasta_res50
# from lib.dataset.HICO_dataset import HICO_Testset
from lib.ult.config import cfg
# from lib.ult.ult import Get_next_sp_with_pose
from train_pasta_net import load_model, saveWeight

from lib.networks.resnet_v1.resnetv1_torch import resnet50 as resnet50_v1

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', 
            help='the path of tf weight to load from',
            default="",
            type=str)
    parser.add_argument('--saveH5', 
            help='save H5 file or not',
            default=True,
            type=bool)
    args = parser.parse_args()
    return args

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights

'''
    load resnet_v1 50
    
    Batchnorm layer:
        tf.gamma == torch.weight
        tf.beta  == torch.bias
'''
def resnet50LoadWeightFromTF(resnet50, pretrained_dict):
    
    # conv1
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['resnet_v1_50/conv1/weights'])
    resnet50.conv1.load_state_dict(tmp_dict)

    # bn1
    tmp_dict = {}
    prefix   = "resnet_v1_50/"
    tmp_dict['weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])
    resnet50.bn1.load_state_dict(tmp_dict)

    '''
        loading resnet_v1 50
        torch.layer1 -> tf.block1
    '''
    tmp_dict = {}

    prefix   = "resnet_v1_50/block1/unit_1/bottleneck_v1/"
    tmp_dict['0.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['0.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['0.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['0.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['0.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['0.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['0.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['0.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['0.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['0.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['0.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['0.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['0.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['0.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['0.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    tmp_dict['0.downsample.0.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/weights'])
    tmp_dict['0.downsample.1.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/gamma'])
    tmp_dict['0.downsample.1.bias']         = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/beta'])
    tmp_dict['0.downsample.1.running_mean'] = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_mean'])
    tmp_dict['0.downsample.1.running_var']  = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block1/unit_2/bottleneck_v1/"
    tmp_dict['1.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['1.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['1.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['1.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['1.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['1.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['1.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['1.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['1.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['1.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['1.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['1.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['1.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['1.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['1.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])


    prefix = "resnet_v1_50/block1/unit_3/bottleneck_v1/"
    tmp_dict['2.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['2.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['2.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['2.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['2.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['2.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['2.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['2.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['2.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['2.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['2.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['2.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['2.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['2.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['2.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    # load layer1
    resnet50.layer1.load_state_dict(tmp_dict)

    '''
        loading resnet_v1 50
        torch.layer2 -> tf.block2
    '''

    prefix   = "resnet_v1_50/block2/unit_1/bottleneck_v1/"
    tmp_dict = {}
    tmp_dict['0.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['0.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['0.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['0.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['0.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['0.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['0.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['0.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['0.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['0.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['0.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['0.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['0.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['0.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['0.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    tmp_dict['0.downsample.0.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/weights'])
    tmp_dict['0.downsample.1.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/gamma'])
    tmp_dict['0.downsample.1.bias']         = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/beta'])
    tmp_dict['0.downsample.1.running_mean'] = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_mean'])
    tmp_dict['0.downsample.1.running_var']  = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block2/unit_2/bottleneck_v1/"
    tmp_dict['1.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['1.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['1.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['1.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['1.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['1.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['1.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['1.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['1.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['1.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['1.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['1.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['1.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['1.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['1.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block2/unit_3/bottleneck_v1/"
    tmp_dict['2.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['2.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['2.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['2.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['2.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['2.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['2.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['2.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['2.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['2.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['2.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['2.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['2.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['2.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['2.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block2/unit_4/bottleneck_v1/"
    tmp_dict['3.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['3.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['3.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['3.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['3.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['3.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['3.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['3.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['3.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['3.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['3.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['3.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['3.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['3.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['3.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    # load layer2
    resnet50.layer2.load_state_dict(tmp_dict)


    '''
        loading resnet_v1 50
        torch.layer3 -> tf.block3
    '''

    prefix   = "resnet_v1_50/block3/unit_1/bottleneck_v1/"
    tmp_dict = {}
    tmp_dict['0.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['0.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['0.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['0.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['0.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['0.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['0.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['0.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['0.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['0.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['0.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['0.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['0.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['0.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['0.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    tmp_dict['0.downsample.0.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/weights'])
    tmp_dict['0.downsample.1.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/gamma'])
    tmp_dict['0.downsample.1.bias']         = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/beta'])
    tmp_dict['0.downsample.1.running_mean'] = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_mean'])
    tmp_dict['0.downsample.1.running_var']  = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block3/unit_2/bottleneck_v1/"
    tmp_dict['1.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['1.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['1.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['1.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['1.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['1.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['1.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['1.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['1.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['1.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['1.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['1.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['1.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['1.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['1.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block3/unit_3/bottleneck_v1/"
    tmp_dict['2.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['2.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['2.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['2.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['2.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['2.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['2.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['2.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['2.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['2.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['2.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['2.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['2.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['2.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['2.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block3/unit_4/bottleneck_v1/"
    tmp_dict['3.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['3.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['3.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['3.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['3.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['3.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['3.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['3.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['3.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['3.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['3.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['3.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['3.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['3.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['3.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block3/unit_5/bottleneck_v1/"
    tmp_dict['4.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['4.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['4.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['4.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['4.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['4.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['4.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['4.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['4.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['4.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['4.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['4.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['4.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['4.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['4.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block3/unit_6/bottleneck_v1/"
    tmp_dict['5.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['5.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['5.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['5.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['5.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['5.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['5.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['5.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['5.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['5.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['5.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['5.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['5.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['5.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['5.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    # load layer3
    resnet50.layer3.load_state_dict(tmp_dict)


    '''
        loading resnet_v1 50
        torch.layer4 -> tf.block4
    '''

    prefix   = "resnet_v1_50/block4/unit_1/bottleneck_v1/"
    tmp_dict = {}
    tmp_dict['0.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['0.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['0.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['0.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['0.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['0.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['0.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['0.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['0.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['0.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['0.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['0.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['0.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['0.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['0.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    tmp_dict['0.downsample.0.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/weights'])
    tmp_dict['0.downsample.1.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/gamma'])
    tmp_dict['0.downsample.1.bias']         = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/beta'])
    tmp_dict['0.downsample.1.running_mean'] = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_mean'])
    tmp_dict['0.downsample.1.running_var']  = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block4/unit_2/bottleneck_v1/"
    tmp_dict['1.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['1.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['1.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['1.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['1.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['1.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['1.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['1.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['1.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['1.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['1.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['1.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['1.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['1.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['1.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block4/unit_3/bottleneck_v1/"
    tmp_dict['2.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['2.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['2.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['2.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['2.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['2.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['2.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['2.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['2.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['2.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['2.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['2.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['2.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['2.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['2.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    # load layer4
    resnet50.layer4.load_state_dict(tmp_dict)


    '''
        loading resnet_v1 50
        torch.layer5 -> tf.block5
    '''

    prefix   = "resnet_v1_50/block5/unit_1/bottleneck_v1/"
    tmp_dict = {}
    tmp_dict['0.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['0.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['0.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['0.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['0.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['0.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['0.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['0.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['0.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['0.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['0.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['0.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['0.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['0.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['0.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    tmp_dict['0.downsample.0.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/weights'])
    tmp_dict['0.downsample.1.weight']       = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/gamma'])
    tmp_dict['0.downsample.1.bias']         = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/beta'])
    tmp_dict['0.downsample.1.running_mean'] = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_mean'])
    tmp_dict['0.downsample.1.running_var']  = torch.Tensor(pretrained_dict[prefix + 'shortcut/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block5/unit_2/bottleneck_v1/"
    tmp_dict['1.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['1.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['1.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['1.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['1.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['1.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['1.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['1.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['1.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['1.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['1.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['1.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['1.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['1.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['1.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    prefix = "resnet_v1_50/block5/unit_3/bottleneck_v1/"
    tmp_dict['2.conv1.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv1/weights'])
    tmp_dict['2.bn1.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/gamma'])
    tmp_dict['2.bn1.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/beta'])
    tmp_dict['2.bn1.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_mean'])
    tmp_dict['2.bn1.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv1/BatchNorm/moving_variance'])

    tmp_dict['2.conv2.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv2/weights'])
    tmp_dict['2.bn2.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/gamma'])
    tmp_dict['2.bn2.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/beta'])
    tmp_dict['2.bn2.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_mean'])
    tmp_dict['2.bn2.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv2/BatchNorm/moving_variance'])

    tmp_dict['2.conv3.weight']      = torch.Tensor(pretrained_dict[prefix + 'conv3/weights'])
    tmp_dict['2.bn3.weight']        = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/gamma'])
    tmp_dict['2.bn3.bias']          = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/beta'])
    tmp_dict['2.bn3.running_mean']  = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_mean'])
    tmp_dict['2.bn3.running_var']   = torch.Tensor(pretrained_dict[prefix + 'conv3/BatchNorm/moving_variance'])

    # load layer5
    resnet50.layer5.load_state_dict(tmp_dict)

    return resnet50

def customLoadWeightFromTF(net, pretrained_dict):

    tmp_dict = {}
    tmp_dict['0.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected/weights'])
    tmp_dict['0.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected/biases'])
    tmp_dict['0.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected_1/weights'])
    tmp_dict['0.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected_1/biases'])
    tmp_dict['1.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected/weights'])
    tmp_dict['1.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected/biases'])
    tmp_dict['1.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected_1/weights'])
    tmp_dict['1.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected_1/biases'])
    tmp_dict['2.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected/weights'])
    tmp_dict['2.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected/biases'])
    tmp_dict['2.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected_1/weights'])
    tmp_dict['2.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected_1/biases'])
    tmp_dict['3.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected/weights'])
    tmp_dict['3.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected/biases'])
    tmp_dict['3.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected_1/weights'])
    tmp_dict['3.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected_1/biases'])
    tmp_dict['4.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected/weights'])
    tmp_dict['4.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected/biases'])
    tmp_dict['4.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected_1/weights'])
    tmp_dict['4.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected_1/biases'])
    tmp_dict['5.0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected/weights'])
    tmp_dict['5.0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected/biases'])
    tmp_dict['5.3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected_1/weights'])
    tmp_dict['5.3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected_1/biases'])
    net.fc7_parts.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected_2/weights'])
    tmp_dict['0.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_0/fully_connected_2/biases'])
    tmp_dict['1.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected_2/weights'])
    tmp_dict['1.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_1/fully_connected_2/biases'])
    tmp_dict['2.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected_2/weights'])
    tmp_dict['2.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_2/fully_connected_2/biases'])
    tmp_dict['3.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected_2/weights'])
    tmp_dict['3.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_3/fully_connected_2/biases'])
    tmp_dict['4.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected_2/weights'])
    tmp_dict['4.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_4/fully_connected_2/biases'])
    tmp_dict['5.weight'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected_2/weights'])
    tmp_dict['5.bias'] = torch.Tensor(pretrained_dict['pasta_classification/cls_pasta_5/fully_connected_2/biases'])
    net.part_cls_scores.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected/weights'])
    tmp_dict['0.bias'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected/biases'])
    tmp_dict['3.weight'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected_1/weights'])
    tmp_dict['3.bias'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected_1/biases'])
    net.fc9_vec.load_state_dict(tmp_dict)
    
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected_2/weights'])
    tmp_dict['bias'] = torch.Tensor(pretrained_dict['vec_classification/fully_connected_2/biases'])
    net.vec_cls_scores.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['vec_attention/fully_connected/weights'])
    tmp_dict['0.bias'] = torch.Tensor(pretrained_dict['vec_attention/fully_connected/biases'])
    net.fc7_P_att.load_state_dict(tmp_dict)
    
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['verb_classification/fully_connected/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['verb_classification/fully_connected/biases'])
    net.verb_cls_scores.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['region_classification/cls_score_P/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['region_classification/cls_score_P/biases'])
    net.P_cls_scores.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['region_classification/cls_score_A/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['region_classification/cls_score_A/biases'])
    net.A_cls_scores.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['region_classification/fully_connected/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['region_classification/fully_connected/biases'])
    net.fc8_L.load_state_dict(tmp_dict)

    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['region_classification/cls_score_L/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['region_classification/cls_score_L/biases'])
    net.L_cls_scores.load_state_dict(tmp_dict)

    return net

if __name__ == "__main__":
    # temp settings
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # arg parsing
    args = parse_args()
    np.random.seed(cfg.RNG_SEED)

    # change tensorflow ckpt to h5 file
    # NHWC -> HCHW
    CKPT_PATH = args.ckpt
    ckpt_dir, ckpt_file_name = os.path.split(CKPT_PATH)
    ckpt_file_name = os.path.splitext(ckpt_file_name)[0]
    H5_PATH   = os.path.join(ckpt_dir, ckpt_file_name + '.h5')
    out_ckpt_path  = os.path.join(ckpt_dir, ckpt_file_name + '_to_pytorch.tar')

    print('Reading TF Weights...')
    weights = read_ckpt(CKPT_PATH)
    print("Finished.")
    keys = sorted(list(weights.keys()))

    # for each_key in keys:
    #     print(each_key, weights[each_key].shape)
    # sys.exit()

    if args.saveH5:
        dd.io.save(H5_PATH, weights)
        print("Saved H5 weight to: ", H5_PATH)

    pretrained_dict = dd.io.load(H5_PATH)
    
    print("Constructing PaStaNet...")
    net = pasta_res50()
    print("Finished.")

    # for k, v in net.state_dict().items():
    #     print(k, v.shape)
    # sys.exit()

    # print("Loading PaSta weight...")
    # net = customLoadWeightFromTF(net, pretrained_dict)
    # print("Finished.")

    # resnet v1
    resnet50 = resnet50_v1()
    resnet50.layer5 = copy.deepcopy(resnet50.layer4)

    print("Loading ResNet50 weight...")
    resnet50 = resnet50LoadWeightFromTF(resnet50, pretrained_dict)
    print("Finished.")

    # using pretrained resnet to TIN net
    resnet50.conv1.padding = 0
    net.resnet50 = resnet50 
    net.image_to_head = nn.Sequential(
            nn.ConstantPad2d((0, 0, 3, 3), 0),
            nn.ConstantPad2d((3, 3, 0, 0), 0),
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            nn.ConstantPad2d((0, 0, 1, 1), 0),
            nn.ConstantPad2d((1, 1, 0, 0), 0),
            nn.MaxPool2d(kernel_size=[3, 3], stride=2),
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3 )

    net.resnet_layer4 = resnet50.layer4
    net.resnet_layer5 = resnet50.layer5

    # save to torch weight file
    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    epoch = 0
    loss  = -1
    
    saveWeight(net, optimizer, epoch, loss, out_ckpt_path)
    print("Saved pytorch weight to: ", out_ckpt_path)

    