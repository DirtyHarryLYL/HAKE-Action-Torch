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
 
from lib.networks.TIN_HICO import TIN_ResNet50
from lib.dataset.HICO_dataset import HICO_Testset
from lib.ult.config_TIN import cfg
from lib.ult.ult import Get_next_sp_with_pose
from train_HICO import load_model, saveWeight

from lib.networks.resnet_v1.resnetv1_torch import resnet50 as resnet50_v1

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', 
            help='the path of tf weight to load from',
            default="/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Weight/TIN_HICO/HOI_iter_1700000.ckpt",
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

if __name__ == "__main__":
    # temp settings
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # arg parsing
    args = parse_args()
    np.random.seed(cfg.RNG_SEED)

    # change tensorflow ckpt to h5 file
    # NHWC -> HCHW
    CKPT_PATH = args.ckpt
    H5_PATH   = args.ckpt.strip('.ckpt') + '.h5'

    weights = read_ckpt(CKPT_PATH)

    if args.saveH5:
        dd.io.save(H5_PATH, weights)
        print("==> save H5 weight to: ", H5_PATH)

    pretrained_dict = dd.io.load(H5_PATH)

    net = TIN_ResNet50()
    '''
    for k, v in net.state_dict().items():
        print(k, v.shape)
    '''

    # sp_to_head
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['resnet_v1_50/conv1_sp/weights'])
    tmp_dict['0.bias'] = torch.Tensor(pretrained_dict['resnet_v1_50/conv1_sp/biases'])
    tmp_dict['3.weight'] = torch.Tensor(pretrained_dict['resnet_v1_50/conv2_sp/weights'])
    tmp_dict['3.bias'] = torch.Tensor(pretrained_dict['resnet_v1_50/conv2_sp/biases'])
    net.sp_to_head.load_state_dict(tmp_dict)

    # head_phi
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['head_phi/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['head_phi/biases'])
    net.head_phi.load_state_dict(tmp_dict)

    # head_g
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['head_g/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['head_g/biases'])
    net.head_g.load_state_dict(tmp_dict)

    # attention_pool_layer_H_network
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['Att_H/fc1_b/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['Att_H/fc1_b/biases'])
    net.attention_pool_layer_H_network.load_state_dict(tmp_dict)

    # attention_pool_layer_O_network
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['Att_O/fc1_b/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['Att_O/fc1_b/biases'])
    net.attention_pool_layer_O_network.load_state_dict(tmp_dict)

    # head_bottleneck
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['bottleneck/bottleneck/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['bottleneck/bottleneck/biases'])
    net.head_bottleneck.load_state_dict(tmp_dict)

    # fc8_SH
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['fc8_SH/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['fc8_SH/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['fc8_SH/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['fc8_SH/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['fc8_SH/BatchNorm/moving_variance'])
    net.fc8_SH.load_state_dict(tmp_dict)

    # fc9_SH
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['fc9_SH/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['fc9_SH/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['fc9_SH/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['fc9_SH/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['fc9_SH/BatchNorm/moving_variance'])
    net.fc9_SH.load_state_dict(tmp_dict)

    # fc8_SO
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['fc8_SO/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['fc8_SO/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['fc8_SO/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['fc8_SO/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['fc8_SO/BatchNorm/moving_variance'])
    net.fc8_SO.load_state_dict(tmp_dict)

    # fc9_SO
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['fc9_SO/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['fc9_SO/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['fc9_SO/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['fc9_SO/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['fc9_SO/BatchNorm/moving_variance'])
    net.fc9_SO.load_state_dict(tmp_dict)

    # Concat_SHsp
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['Concat_SHsp/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['Concat_SHsp/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['Concat_SHsp/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['Concat_SHsp/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['Concat_SHsp/BatchNorm/moving_variance'])
    net.Concat_SHsp.load_state_dict(tmp_dict)

    # fc7_SHsp
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['fc7_SHsp/weights'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['fc7_SHsp/BatchNorm/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['fc7_SHsp/BatchNorm/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['fc7_SHsp/BatchNorm/moving_mean'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['fc7_SHsp/BatchNorm/moving_variance'])
    net.fc7_SHsp.load_state_dict(tmp_dict)

    # pool2_flat_pose_map
    tmp_dict = {}
    tmp_dict['0.weight'] = torch.Tensor(pretrained_dict['fc_binary/conv1_pose_map/weights'])
    tmp_dict['0.bias']   = torch.Tensor(pretrained_dict['fc_binary/conv1_pose_map/biases'])
    tmp_dict['3.weight'] = torch.Tensor(pretrained_dict['fc_binary/conv2_pose_map/weights'])
    tmp_dict['3.bias']   = torch.Tensor(pretrained_dict['fc_binary/conv2_pose_map/biases'])
    net.pool2_flat_pose_map.load_state_dict(tmp_dict)
    
    # fc8_binary_1
    dict_fc8_binary_1 = {}
    dict_fc8_binary_1['0.weight'] = torch.Tensor(pretrained_dict['fc_binary/fc8_binary_1/weights'])
    dict_fc8_binary_1['0.bias']   = torch.Tensor(pretrained_dict['fc_binary/fc8_binary_1/biases'])
    net.fc8_binary_1.load_state_dict(dict_fc8_binary_1)

    # fc8_binary_2
    dict_fc8_binary_2 = {}
    dict_fc8_binary_2['0.weight'] = torch.Tensor(pretrained_dict['fc_binary/fc8_binary_2/weights'])
    dict_fc8_binary_2['0.bias']   = torch.Tensor(pretrained_dict['fc_binary/fc8_binary_2/biases'])
    net.fc8_binary_2.load_state_dict(dict_fc8_binary_2)

    # fc9_binary
    dict_fc9_binary = {}
    dict_fc9_binary['0.weight'] = torch.Tensor(pretrained_dict['fc_binary/fc9_binary/weights'])
    dict_fc9_binary['0.bias']   = torch.Tensor(pretrained_dict['fc_binary/fc9_binary/biases'])
    net.fc9_binary.load_state_dict(dict_fc9_binary)

    # cls_score_H
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['classification/cls_score_H/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['classification/cls_score_H/biases'])
    net.cls_score_H.load_state_dict(tmp_dict)

    # cls_score_O
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['classification/cls_score_O/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['classification/cls_score_O/biases'])
    net.cls_score_O.load_state_dict(tmp_dict)

    # cls_score_sp
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['classification/cls_score_sp/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['classification/cls_score_sp/biases'])
    net.cls_score_sp.load_state_dict(tmp_dict)

    # cls_score_binary
    tmp_dict = {}
    tmp_dict['weight'] = torch.Tensor(pretrained_dict['binary_classification/cls_score_binary/weights'])
    tmp_dict['bias']   = torch.Tensor(pretrained_dict['binary_classification/cls_score_binary/biases'])
    net.cls_score_binary.load_state_dict(tmp_dict)

    # resnet v2 (implement by torch)
    # resnet50 = torch.hub.load('pytorch/vision:v0.4.0', 'resnet50', pretrained=False)
    
    # resnet v1
    resnet50 = resnet50_v1()
    resnet50.layer5 = copy.deepcopy(resnet50.layer4)

    print("==> loading weight for resnet")
    resnet50 = resnet50LoadWeightFromTF(resnet50, pretrained_dict)
    print("==> finish loading")

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
    ckp_path = cfg.WEIGHT_DIR + "BestweightLoadFromTf-resV1-addBn-Ckpt.tar"
    
    saveWeight(net, optimizer, epoch, loss, ckp_path)
    print("==> save torch weight to: ", ckp_path)

    