#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Oct 22th, 2020            #
#############################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import lmdb
import copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from lib.networks.pasta_net import pasta_res50
from lib.dataset.hake_dataset import hake_test_pasta
from lib.ult.config import cfg
from ult.ult_Generalized import Generate_relation_bbox, Generate_action_object, obj2str, str2obj, custom_collate_fn
import os.path as osp
from utils.custom_multiprocessing import process_pool

def parse_args():
    parser = argparse.ArgumentParser(description='Test PaStaNet')
    parser.add_argument('--weight', 
            help='the path of weight to load from',
            default="",
            type=str)
    parser.add_argument('--eval', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    parser.add_argument('--benchmark', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    args = parser.parse_args()
    return args

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def test_net(net, test_loader, output_dir):
    np.random.seed(cfg.RNG_SEED)
    for test_data in tqdm(test_loader):

        image, dataset_name, im_name, test_rcnn = test_data['image'], test_data['dataset'][0], test_data['im_name'][0], test_data['test_rcnn'][0]
        out_path = osp.join(output_dir, im_name+'.pkl')
        if not os.path.exists(out_path):
            keys, bboxes, pasta_scores_list, verb_scores_list = [], [], [], []
            blobs        = {}
            blobs['image'] = image  
            blobs['dataset'] = test_data['dataset'] 
            if dataset_name == 'collect':
                for i in range(len(test_rcnn)):
                    Human_out = test_rcnn[i]
                    H_box = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
                    P_box = Human_out[8][None, ...]
                    blobs['H_num'] = 1
                    blobs['H_boxes'] = H_box
                    blobs['P_boxes'] = P_box
                    for k, v in blobs.items():
                        if k in ['image_id', 'H_num', 'image', 'dataset']:
                            continue
                        blobs[k] = torch.from_numpy(v)

                    with torch.no_grad():
                        _, pasta_scores, verb_scores = net(blobs, mode='inference')
                    pasta_scores = pasta_scores.cpu().detach().numpy()
                    verb_scores = verb_scores.cpu().detach().numpy()
                    torch.cuda.empty_cache()
                    keys.append([dataset_name, im_name])
                    hbox = np.array(Human_out[2]).reshape(1, -1)
                    obox = np.array(Human_out[2]).reshape(1, -1)
                    bboxes.append(np.concatenate([hbox, obox], axis=1))
                    pasta_scores_list.append(pasta_scores)
                    verb_scores_list.append(verb_scores)
            else:
                for i in range(len(test_rcnn)):
                    Human_out = test_rcnn[i]
                    if Human_out[2] == 1:
                        H_box = np.array([0, Human_out[1][0],  Human_out[1][1],  Human_out[1][2],  Human_out[1][3]]).reshape(1,5)
                        P_box = Human_out[6][None, ...]

                        blobs['H_num'] = 1
                        blobs['H_boxes'] = H_box
                        blobs['P_boxes'] = P_box

                        for k, v in blobs.items():
                            if k in ['image_id', 'H_num', 'image', 'dataset']:
                                continue
                            blobs[k] = torch.from_numpy(v)
                        with torch.no_grad():
                            _, pasta_scores, verb_scores = net(blobs, mode='inference')
                        pasta_scores = pasta_scores.cpu().detach().numpy()
                        verb_scores = verb_scores.cpu().detach().numpy()
                        torch.cuda.empty_cache()
                        keys.append([dataset_name, im_name])
                        hbox = np.array(Human_out[1]).reshape(1, -1)
                        obox = np.array(Human_out[1]).reshape(1, -1)
                        bboxes.append(np.concatenate([hbox, obox], axis=1))
                        pasta_scores_list.append(pasta_scores)
                        verb_scores_list.append(verb_scores)
                        
                pickle.dump([keys, bboxes, pasta_scores_list, verb_scores_list], open(out_path, 'wb'))
                
if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    args.pasta_mode = 2
    args.pasta_trained = '-1'
    cfg.WEIGHT_PATH  = args.weight
    model_dir, model_filename = os.path.split(args.weight)
    model_name = os.path.split(model_dir)[-1]
    output_dir = os.path.join(cfg.ROOT_DIR, '-Results', model_name, model_filename+'_results')

    if args.eval:
        print("Testing PaStaNet, output_dir: ", output_dir, '...')
        os.makedirs(output_dir, exist_ok=True)

        # model preparing
        net = pasta_res50(args)
        net = load_model(net, cfg.WEIGHT_PATH)
        net.testMode = True
        net.eval()
        net = net.cuda()
        
        # data preparing
        Test_path_hico   = os.path.join(cfg.DATA_DIR, 'Test_all_part_lmdb')
        Test_path_40v    = os.path.join(cfg.DATA_DIR, 'hake_40v_test_gt_lmdb')
        test_set    = hake_test_pasta(Test_path_hico, Test_path_40v)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        
        test_net(net, test_loader, output_dir)
    
    if args.benchmark:
        # final benchmark
        benchmark_pool = process_pool()
        print('Evaluating...')
        cmd = 'python -u {binary} --result-dir {model} '

        cmd = cmd.format(binary='Evaluate_HAKE_93.py',
                        model=output_dir)
        cmd_cwd_list = [
                            (cmd,'./-Results')
                    ]

        benchmark_pool.apply(cmd_cwd_list)
        benchmark_pool.wait()
        benchmark_pool.reset()
        cmd_cwd_list = []
        print('Finished.')
