#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Oct 22th, 2020            #
#############################################
import argparse
import os
import os.path as osp
import sys
import json
import pickle
import numpy as np
import cv2
import torch
import ipdb
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__),'..','utils')))
from custom_multiprocessing import process_pool
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__),'..','lib')))
from networks.pasta_net import pasta_res50
from ult.config import cfg

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

class pasta_model():

    def __init__(self, weight_path):
        super(pasta_model, self).__init__()
        self.weight_path = weight_path
        self.context_builder()
    
    # build pasta context
    def context_builder(self):
        self.net = pasta_res50(None)
        self.net = load_model(self.net, self.weight_path)
        self.net.testMode = True
        self.net.eval()
        self.net = self.net.cuda()

    # build preprocessor
    def data_preprocess(self, data, data_type='box'):
        in_data = data.float()
        if data_type == 'image':
            return in_data.unsqueeze(0)
        elif data_type == 'box':
            in_data = in_data.unsqueeze(0)
            zeros_suffix = torch.zeros((1, 1)).float()
            out_data = torch.cat([zeros_suffix, in_data], 1)
            return out_data
        else:
            return data

    # collect data
    def data_builder(self, im_path, box, part_boxes):
        im       = cv2.imread(im_path)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_orig  = im_orig.transpose(2, 0, 1)

        out_data = {}
        out_data['image'] = self.data_preprocess(torch.from_numpy(im_orig), 'image')
        out_data['H_boxes'] = self.data_preprocess(torch.from_numpy(box), 'box')
        out_data['O_boxes'] = out_data['H_boxes']
        out_data['R_boxes'] = out_data['H_boxes']
        out_data['P_boxes'] = torch.from_numpy(part_boxes).unsqueeze(0)
        out_data['H_num'] = 1
        out_data['gt_object'] = torch.from_numpy(np.ones((1, 80), dtype=np.float32))
        return out_data
        
    def inference(self, alpha_data):
        im_path, box, part_boxes = alpha_data['image_id'], alpha_data['box'], alpha_data['part_boxes']
        blob = self.data_builder(im_path, box, part_boxes)
        results = list(self.net(blob, mode='inference'))
        for idx in range(len(results)):
            results[idx] = results[idx].cpu().detach().numpy()
        return results