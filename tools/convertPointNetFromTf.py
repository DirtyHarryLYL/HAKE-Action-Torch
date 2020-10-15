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
import os
import sys
import inspect
import deepdish as dd
import copy
from   tqdm import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from hakeaction.models.DJRN import DJRN_ResNet50
from hakeaction.configs.config_DJRN import cfg
from hakeaction.core.backbones.resnet_v1.resnetv1_torch import resnet50 as resnet50_v1
from hakeaction.core.utils.pointnet import *
from train_HICO import saveWeight

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader    = tf.train.NewCheckpointReader(ckpt)
    weights   = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights


def pointNetLoadWeightFromTF(pointNet, pretrained_dict):
    
    # conv1
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv1/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv1/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv1/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv1/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze/ExponentialMovingAverage']) 
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    

    print(tmp_dict['1.running_mean'].shape)
    print(tmp_dict['1.running_var'].shape)

    data1 = tf.Variable(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze/ExponentialMovingAverage'])
    data2 = tf.Variable(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    moments_dims   = [0]
    mean, variance = tf.nn.moments(data1, moments_dims)
    print(mean, variance)
    
    # '''
    batch_mean   = tf.Variable(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze/ExponentialMovingAverage'])
    batch_var    = tf.Variable(pretrained_dict['conv1/bn/conv1/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    ema          = tf.train.ExponentialMovingAverage(decay=0.9)
    is_training  = tf.constant(False)
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mean = mean.eval(session=sess)
    var  = var.eval(session=sess)

    tmp_dict['1.running_mean']  = torch.Tensor(mean) 
    tmp_dict['1.running_var']   = torch.Tensor(var)
    # '''

    pointNet.conv1.load_state_dict(tmp_dict)

    # conv2
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv2/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv2/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv2/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv2/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv2/bn/conv2/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv2/bn/conv2/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv2.load_state_dict(tmp_dict)

    # tconv1
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv1/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv1/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv1/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv1/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['transform_net2/tconv1/bn/transform_net2/tconv1/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['transform_net2/tconv1/bn/transform_net2/tconv1/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.tconv1.load_state_dict(tmp_dict)

    # tconv2
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv2/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv2/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv2/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv2/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['transform_net2/tconv2/bn/transform_net2/tconv2/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['transform_net2/tconv2/bn/transform_net2/tconv2/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.tconv2.load_state_dict(tmp_dict)

    # tconv3
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv3/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv3/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['transform_net2/tconv3/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['transform_net2/tconv3/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['transform_net2/tconv3/bn/transform_net2/tconv3/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['transform_net2/tconv3/bn/transform_net2/tconv3/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.tconv3.load_state_dict(tmp_dict)

    # tfc1
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['transform_net2/tfc1/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['transform_net2/tfc1/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['transform_net2/tfc1/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['transform_net2/tfc1/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['transform_net2/tfc1/bn/transform_net2/tfc1/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['transform_net2/tfc1/bn/transform_net2/tfc1/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.tfc1.load_state_dict(tmp_dict)

    # tfc2
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['transform_net2/tfc2/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['transform_net2/tfc2/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['transform_net2/tfc2/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['transform_net2/tfc2/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['transform_net2/tfc2/bn/transform_net2/tfc2/bn/moments/Squeeze/ExponentialMovingAverage'])
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['transform_net2/tfc2/bn/transform_net2/tfc2/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.tfc2.load_state_dict(tmp_dict)

    # weights and biases
    pointNet.weights = torch.Tensor(pretrained_dict['transform_net2/transform_feat/weights']).view(256, 4096)  
    pointNet.biases  = torch.Tensor(pretrained_dict['transform_net2/transform_feat/biases']) 

    # conv3
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv3/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv3/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv3/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv3/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv3/bn/conv3/bn/moments/Squeeze/ExponentialMovingAverage']) # TODO Check
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv3/bn/conv3/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv3.load_state_dict(tmp_dict)

    # conv4
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv4/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv4/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv4/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv4/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv4/bn/conv4/bn/moments/Squeeze/ExponentialMovingAverage']) # TODO Check
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv4/bn/conv4/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv4.load_state_dict(tmp_dict)

    # conv5
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv5/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv5/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv5/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv5/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv5/bn/conv5/bn/moments/Squeeze/ExponentialMovingAverage']) # TODO Check
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv5/bn/conv5/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv5.load_state_dict(tmp_dict)

    # conv6
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv6/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv6/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv6/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv6/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv6/bn/conv6/bn/moments/Squeeze/ExponentialMovingAverage']) # TODO Check
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv6/bn/conv6/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv6.load_state_dict(tmp_dict)

    # conv7
    tmp_dict = {}
    tmp_dict['0.weight']        = torch.Tensor(pretrained_dict['conv7/weights'])
    tmp_dict['0.bias']          = torch.Tensor(pretrained_dict['conv7/biases'])
    tmp_dict['1.weight']        = torch.Tensor(pretrained_dict['conv7/bn/gamma'])
    tmp_dict['1.bias']          = torch.Tensor(pretrained_dict['conv7/bn/beta'])
    tmp_dict['1.running_mean']  = torch.Tensor(pretrained_dict['conv7/bn/conv7/bn/moments/Squeeze/ExponentialMovingAverage']) # TODO Check
    tmp_dict['1.running_var']   = torch.Tensor(pretrained_dict['conv7/bn/conv7/bn/moments/Squeeze_1/ExponentialMovingAverage'])
    pointNet.conv7.load_state_dict(tmp_dict)

    return pointNet

if __name__ == "__main__":
    # arg parsing
    np.random.seed(cfg.RNG_SEED)

    # change tensorflow ckpt to h5 file
    # '''
    CKPT_PATH       = "/home/zhanke/DJ-RN/DJ-RN.torch/tmp/tf_pointNet_ckpt/model_10000.ckpt"
    H5_PATH         = CKPT_PATH.strip('.ckpt') + '.h5'
    pretrained_dict = read_ckpt(CKPT_PATH)

    # load pretrained weight for pointNet_torch
    pointNet_torch = PointNetHico(K=64, )
    pointNet_torch = pointNetLoadWeightFromTF(pointNet_torch, pretrained_dict)
    pointNet_torch.eval()

    # DEBUG
    data = pickle.load(open('./tmp/pointNet_debug.pkl', 'rb'), encoding="bytes")
    '''
    conv1, transform, point_feat, global_feat, global_feat_expand, concat_feat, net
    (1, 1228, 1, 64)
    (1, 64, 64)
    (1, 1228, 1, 64)
    (1, 1, 1, 1024)
    (1, 1228, 1, 1024)
    (1, 1228, 1, 1088)
    (1, 1228, 1, 256)
    '''

    # test_input     = torch.randn(1, 1, 1228, 3) 
    tf_conv1              = torch.Tensor(data[b'output'][0][0]).permute(0,3,1,2)
    tf_transform          = torch.Tensor(data[b'output'][0][1])
    tf_point_feat         = torch.Tensor(data[b'output'][0][2]).permute(0,3,1,2)
    tf_global_feat        = torch.Tensor(data[b'output'][0][3]).permute(0,3,1,2)
    tf_global_feat_expand = torch.Tensor(data[b'output'][0][4]).permute(0,3,1,2)
    tf_concat_feat        = torch.Tensor(data[b'output'][0][5]).permute(0,3,1,2)
    tf_output             = torch.Tensor(data[b'output'][0][6]).permute(0,3,1,2)
    tf_input              = torch.Tensor(data[b'input']).unsqueeze(0)
    
    conv1, transform, point_feat, global_feat, global_feat_expand, concat_feat, torch_output = pointNet_torch(tf_input)
    print("tf_input.shape:    ",  tf_input.shape)
    print("conv1:             ",  torch.mean(torch.abs(conv1 - tf_conv1)))
    print("transform:         ",  torch.mean(torch.abs(transform - tf_transform)))
    print("point_feat:        ",  torch.mean(torch.abs(point_feat - tf_point_feat)))
    print("global_feat:       ",  torch.mean(torch.abs(global_feat - tf_global_feat)))
    print("global_feat_expand:",  torch.mean(torch.abs(global_feat_expand - tf_global_feat_expand)))
    print("concat_feat:       ",  torch.mean(torch.abs(concat_feat - tf_concat_feat)))
    print("final output:      ",  torch.mean(torch.abs(torch_output - tf_output)))
