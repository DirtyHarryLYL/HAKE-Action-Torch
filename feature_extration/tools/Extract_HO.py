from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import cPickle as pickle
import os, argparse

from ult.config import cfg
from models.extractor_HO import test_net
from networks.net_HO import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('--data', dest='data',
            help='pkl file to load',
            default='Data/db_trainval_with_pool.pkl', type=str)
    parser.add_argument('--image_path', dest='image_path',
            help='image path',
            default='Data/hico_20160224_det/images/train2015/', type=int)
    parser.add_argument('--output', dest='output',
            help='feature path',
            default='Data/feature/train/', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    db = pickle.load(open(args.data, 'rb'))
    weight = 'Weights/model.ckpt'
  
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, db, args.output, args.image_path)
    sess.close()
    

