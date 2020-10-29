from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import cPickle as pickle
import os, argparse

from ult.config import cfg

from models.extractor_R import test_net as extract_R
from networks.net_R import ResNet50 as net_R

from models.extractor_HO import test_net as extract_HO
from networks.net_HO import ResNet50 as net_HO

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
            default='Data/Union_feature/train/', type=str)
    parser.add_argument('--mode', dest='mode',
            help='Extraction mode, 0 for union feature, otherwise for individual human/object feature',
            default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    db = pickle.load(open(args.data, 'rb'))
  
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    
    test_net, net = None, None
    if args.mode == 0:
        net = net_R()
        test_net = extract_R
    else:
        net = net_HO()
        test_net = extract_HO
        
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, 'Weights/model.ckpt')
    test_net(sess, net, db, args.output, args.image_path)
    sess.close()