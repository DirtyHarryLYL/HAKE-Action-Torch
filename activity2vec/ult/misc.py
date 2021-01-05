#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
import numpy as np
import copy
import torch

def time2str(eta):
    hours = eta // 3600
    minutes = (eta % 3600) // 60
    secs = (eta % 3600) % 60
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, secs)

def loss_reweight(cfg):
    param_k = cfg.TRAIN.LOSS_WEIGHT_K
    pasta_weights = np.load(cfg.DATA.PASTA_WEIGHTS_PATH, allow_pickle=True).item()
    for key in pasta_weights:
        if param_k != 2:
            new_weights = param_k * np.log(param_k * np.exp(pasta_weights[key]/2) / 2)
            new_weights[pasta_weights[key]==0] = 0
            pasta_weights[key] = copy.deepcopy(new_weights)
        pasta_weights[key] = torch.from_numpy(pasta_weights[key]).float().reshape(1, -1).cuda()
    return pasta_weights