#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
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
import copy
from activity2vec.dataset.hake_dataset import draw_relation
from activity2vec.networks.pasta_net import pasta_res50
from activity2vec.ult.config import get_cfg
from easydict import EasyDict as edict
from .part_box_generation import output_part_box, map_17_to_16

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

class pasta_model():

    def __init__(self, cfg):
        super(pasta_model, self).__init__()
        self.cfg = cfg
        self.net = pasta_res50(self.cfg)
        self.net = load_model(self.net, self.cfg.DEMO.A2V_WEIGHT)
        self.net.eval()
        self.net = self.net.cuda()

    # build preprocessor
    def preprocess(self, image, pose_result):
        image    = image.astype(np.float32, copy=True)
        image   -= self.cfg.PIXEL_MEANS
        image    = image.transpose(2, 0, 1)
        image    = torch.from_numpy(image).unsqueeze(0)

        annos = edict()
        annos.human_bboxes = []
        annos.part_bboxes = []
        annos.keypoints = []
        annos.human_scores = []
        annos.skeletons = []
        
        for human in pose_result:
            each_human = copy.deepcopy(human)
            # xywh to xyxy
            each_human['bbox'][2] += each_human['bbox'][0]
            each_human['bbox'][3] += each_human['bbox'][1]
            human_bbox = np.array(each_human['bbox'], dtype=float)
            keypoints = torch.cat([each_human['keypoints'], each_human['kp_score']], 1)
            part_bbox = output_part_box(map_17_to_16(keypoints.numpy()), human_bbox)[:, 1:]
            
            annos.human_bboxes.append(each_human['bbox'])
            annos.part_bboxes.append(part_bbox)
            annos.keypoints.append(keypoints.unsqueeze(0))
            annos.human_scores.append(each_human['proposal_score'].unsqueeze(0))
            annos.skeletons.append(torch.tensor(draw_relation(keypoints[:, :2])).unsqueeze(0))

        annos.human_bboxes = torch.tensor(annos.human_bboxes).unsqueeze(0).float()
        annos.part_bboxes = torch.tensor(annos.part_bboxes).unsqueeze(0).float()
        annos.keypoints = torch.cat(annos.keypoints, 0).unsqueeze(0).float()
        annos.human_scores = torch.cat(annos.human_scores, 0).unsqueeze(0).float()
        annos.skeletons = torch.cat(annos.skeletons, 0).unsqueeze(1).unsqueeze(0).float()
        return image, annos

    def inference(self, image, annos):
        f_pasta, p_pasta, p_verb = self.net(image, annos)
        f_pasta = f_pasta.detach().cpu().numpy()
        p_pasta = p_pasta.detach().cpu().numpy()
        p_verb  = p_verb.detach().cpu().numpy()
        return f_pasta, p_pasta, p_verb