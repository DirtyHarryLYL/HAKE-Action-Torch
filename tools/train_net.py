#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm

from benchmark import benchmark
from test_net import custom_collate, test
from activity2vec.networks.pasta_net import pasta_res50
from activity2vec.dataset.hake_dataset import hake_train, hake_test
from activity2vec.ult.config import get_cfg
from activity2vec.ult.logging import setup_logging
from activity2vec.ult.timer import Timer
from activity2vec.ult.misc import time2str, loss_reweight
from activity2vec.ult.checkpoint import save_model, load_model
from activity2vec.ult.loss import a2v_loss
from activity2vec.ult.parser import parse_args
from activity2vec.ult.visualize import a2v_curve


def train(cfg, net, train_loader, test_loader, optimizer, scheduler, global_iter=0, loggers=None):
    pasta_weights = loss_reweight(cfg)
    data_length = len(train_loader)

    if cfg.TRAIN.SHOW_LOSS_CURVE:
        loss_curve = a2v_curve(cfg)

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        lossPerEpoch    = 0.0
        data_timer = Timer()
        net_timer = Timer()
        global_timer = Timer()
        data_timer.tic()
        global_timer.tic()
        for i, (image, annos) in enumerate(train_loader):

            data_timer.toc()
            net_timer.tic()

            image = image.cuda(non_blocking=True)

            for key in annos:
                if isinstance(annos[key], dict):
                    for sub_key in annos[key]:
                        annos[key][sub_key] = annos[key][sub_key].cuda()
                        annos[key][sub_key] = annos[key][sub_key].squeeze(0)
                else:
                    annos[key] = annos[key].cuda()
                    annos[key] = annos[key].squeeze(0)
            annos['human_bboxes'] = torch.cat([torch.zeros(annos['human_bboxes'].shape[0], 1).cuda(), annos['human_bboxes']], 1)
            annos['part_bboxes'] = torch.cat([torch.zeros(annos['part_bboxes'].shape[0], annos['part_bboxes'].shape[1], 1).cuda(), annos['part_bboxes']], 2)
            s_parts, s_verb = net(image, annos)
            loss = a2v_loss(cfg, s_parts, s_verb, annos, pasta_weights, net.pasta_name2idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net_timer.toc()

            if scheduler is not None:
                scheduler.step()
            
            lossPerEpoch += float(loss)
            mean_loss = float(lossPerEpoch / (i+1))
            global_timer.toc()

            # save weight if necessary
            if global_iter % cfg.TRAIN.CHECKPOINT_INTERVAL == 0:
                loggers.train.info("==> Saving weight: iteration {}".format(global_iter))
                weight_path = os.path.join(cfg.WEIGHT_DIR, "model_{}.pth".format(global_iter))
                save_model(net, optimizer, scheduler, global_iter, weight_path)
                
                if global_iter != 0:
                    # test in training
                    model_dir, model_filename = os.path.split(weight_path)
                    model_name = os.path.split(model_dir)[-1]
                    output_dir = os.path.join(cfg.ROOT_DIR, 'results', model_name, model_filename+'_results')

                    torch.cuda.empty_cache()
                    net.eval()
                    with torch.no_grad():
                        test(cfg, net, test_loader, output_dir, loggers.test)
                        pasta_map, verb_map, map_w_no_interaction_list, map_wo_no_interaction_list = benchmark(output_dir, cfg, loggers.test)
                        
                        if len(cfg.MODEL.MODULE_TRAINED) == 1 and cfg.MODEL.MODULE_TRAINED[0] != 'verb':
                            loss_curve.log({'iteration': global_iter, 
                                            '{:s}_map_w_no_interaction'.format(cfg.MODEL.MODULE_TRAINED[0]): map_w_no_interaction_list[net.pasta_name2idx[cfg.MODEL.MODULE_TRAINED[0]]], 
                                            '{:s}_map_wo_no_interaction'.format(cfg.MODEL.MODULE_TRAINED[0]): map_wo_no_interaction_list[net.pasta_name2idx[cfg.MODEL.MODULE_TRAINED[0]]]})
                        else:
                            loss_curve.log({'iteration': global_iter, 'pasta_map': pasta_map, 'verb_map': verb_map})
                    net.train()
                    torch.cuda.empty_cache()
                    
                    shutil.rmtree(output_dir)
                    
            # show training schedule
            if global_iter % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                out_str = "|iteration: {:6d}|loss: {:.6f}|lr: {:.6f}|time: {:3f}|net_time: {:3f}|data_time: {:3f}|eta: {:s}|".format(global_iter, mean_loss, float(optimizer.param_groups[0]['lr']), global_timer.average_time, net_timer.average_time, data_timer.average_time, time2str(int((len(train_loader) - (i + 1)) * global_timer.average_time)))
                loggers.train.info(out_str)

                if cfg.TRAIN.SHOW_LOSS_CURVE:
                    loss_curve.log({'iteration': global_iter, 'loss': mean_loss})
                    loss_curve.render()

            global_iter += 1

            global_timer.tic()
            data_timer.tic()

def setup():
    # arg parsing
    cfg = get_cfg()
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    assert len(cfg.MODEL_NAME) != 0
    assert cfg.TRAIN.IM_BATCH_SIZE == 1
    assert len(list(set(cfg.TRAIN.DATA_SPLITS) - set(cfg.DATA.FULL_SET_NAMES))) == 0, 'Unknown training split exists!'
    cfg.MODEL_NAME = args.model
    cfg.WEIGHT_DIR = os.path.join(cfg.WEIGHT_DIR, cfg.MODEL_NAME)
    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, cfg.MODEL_NAME)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    cfg.freeze()

    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    with open(os.path.join(cfg.LOG_DIR, 'config.yaml'), "w") as f:
        f.write(cfg.dump())

    return cfg

if __name__ == "__main__":
    
    cfg = setup()

    loggers = setup_logging(cfg.LOG_DIR)
    loggers.train.info('==> Training Activity2Vec <==')
    loggers.train.info('==> cfg:\n' + cfg)

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    train_loader = torch.utils.data.DataLoader(dataset=hake_train(cfg), 
                                               batch_size=cfg.TRAIN.IM_BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=cfg.TRAIN.NUM_WORKERS
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=hake_test(cfg), 
                                              batch_size=1, 
                                              shuffle=False, 
                                              collate_fn=custom_collate, 
                                              num_workers=cfg.TEST.NUM_WORKERS
                                              )

    data_length  = len(train_loader)
    loggers.train.info('==> Data length: {}'.format(data_length))

    # build network
    net   = pasta_res50(cfg)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN.BASE_LR, momentum=0.9)
    if cfg.TRAIN.LR_SCHEDULE == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2*data_length, T_mult=2, eta_min=0)
    elif cfg.TRAIN.LR_SCHEDULE != 'constant':
        raise NotImplementedError
    else:
        scheduler = None

    global_iter = 0
    # load weight if continue training
    if len(cfg.TRAIN.CHECKPOINT_PATH) > 0:
        loggers.train.info("==> Loading weight from: {}".format(cfg.TRAIN.CHECKPOINT_PATH), cfg)
        net, optimizer, scheduler, global_iter = load_model(cfg, net, optimizer, scheduler, cfg.TRAIN.CHECKPOINT_PATH)

    net       = net.cuda()
    train(cfg, net, train_loader, test_loader, optimizer, scheduler, global_iter, loggers)