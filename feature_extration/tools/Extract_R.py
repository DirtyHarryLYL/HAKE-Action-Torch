from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import cPickle as pickle
import os

from ult.config import cfg
from models.extractor_R import test_net
from networks.net_R import ResNet50


if __name__ == '__main__':

    ppdm = pickle.load(open('../data/db_trainval_with_pool.pkl', 'rb'))
    # db_test_feat.pkl
    weight = 'Weights/model.ckpt'
  
    output_file = os.path.join('../data/Union_feature/trainval/')
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, ppdm, output_file)
    sess.close()
    
