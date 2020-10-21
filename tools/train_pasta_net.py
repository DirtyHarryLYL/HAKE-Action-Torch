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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pickle
import argparse
import logging
import time
from tqdm import tqdm
import os
import sys
import inspect
import os.path as osp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from lib.networks.pasta_net import pasta_res50
from lib.dataset.hake_dataset import hake_train
from lib.ult.config import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Training PaStaNet on HAKE-Large')
    parser.add_argument('--model', dest='model',
            help='model name',
            default='', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one.',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='How many ResNet blocks are there?',
            default=5, type=int)
    parser.add_argument('--pasta_mode',
            default=2, type=int)
    parser.add_argument('--pasta_trained',
            default='0,1,2,3,4,5', type=str)
    parser.add_argument('--train_continue', dest='train_continue',
            help='Whether to continue from previous ckpt',
            default=cfg.TRAIN_MODULE_CONTINUE, type=int)
    parser.add_argument('--init_weight', dest='init_weight',
            help='How to init weight',
            default=cfg.TRAIN_INIT_WEIGHT, type=int)
    parser.add_argument('--module_update', dest='module_update',
            help='How to update modules',
            default=cfg.TRAIN_MODULE_UPDATE, type=int)
    parser.add_argument('--train_module', dest='train_module',
            help='How to compute loss',
            default=cfg.TRAIN_MODULE, type=int)
    parser.add_argument('--ckpt_interval', default=20000, type=int)
    parser.add_argument('--base_lr', default=0.0001, type=float)
    parser.add_argument('--weight', 
            help='the path of weight to load from',
            default="",
            type=str)
    parser.add_argument('--lr_schedule', default='constant', type=str)
    parser.add_argument('--load_optim', default=0, type=int)
    args = parser.parse_args()
    return args

def saveWeight(net, optimizer, scheduler, iters, loss, ckp_path):
    torch.save({
                "iters": iters,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': loss,
                }, ckp_path)

def load_model(model, optimizer, scheduler, ckp_path, load_optim):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if 'optimizer_state_dict' in checkpoint and load_optim:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if 'scheduler_state_dict' in checkpoint and load_optim and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for state in scheduler.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    if 'iters' in checkpoint:
        iters = checkpoint['iters']
    else:
        iters = 0
    return model, optimizer, iters

def PRINT_AND_LOG(message):
    print(message)
    with open(osp.join(cfg.LOG_DIR, "train.log"), 'a') as f:
        f.write(message + "\n")

def train(net, train_loader, optimizer, scheduler, global_iter=0, tb_dir='', max_epoch=100, startEpoch=0):
    latestSavedWeight = None
    for epoch in range(startEpoch, max_epoch):
        lossPerEpoch    = 0.0
        for i, blobs in tqdm(enumerate(train_loader)):
            flag = net(blobs)
            total_loss = net.add_loss()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # save weight if necessary
            lossPerEpoch += float(total_loss)
            if global_iter % args.ckpt_interval == 0:
                PRINT_AND_LOG("==> saving weight: iteration {}, loss_per_batch={:.6f}".format(global_iter, float(lossPerEpoch / (i+1))))
                latestSavedWeight = osp.join(cfg.WEIGHT_DIR, "model_{}.pth".format(global_iter))
                saveWeight(net, optimizer, scheduler, global_iter, lossPerEpoch, latestSavedWeight)
            
            # show training schedule
            if global_iter % cfg.TRAIN.DISPLAY == 0:
                out_str = "iteration {}, loss_per_batch={:.6f}, dataset={}, image_id={}, lr={}".format(global_iter, float(lossPerEpoch / (i+1)), blobs['dataset'][0], blobs['image_id'][0], float(optimizer.param_groups[0]['lr']))
                PRINT_AND_LOG(out_str)
                
            global_iter += 1

if __name__ == "__main__":
    # arg parsing
    args = parse_args()

    cfg.TRAIN.SNAPSHOT_ITERS    = args.ckpt_interval
    cfg.TRAIN_MODULE_CONTINUE   = args.train_continue
    cfg.TRAIN_INIT_WEIGHT       = args.init_weight
    cfg.TRAIN_MODULE_UPDATE     = args.module_update
    cfg.TRAIN_MODULE            = args.train_module
    cfg.WEIGHT_PATH             = args.weight
    cfg.WEIGHT_DIR              = osp.join(cfg.WEIGHT_DIR, args.model)
    cfg.LOG_DIR                 = osp.join(cfg.LOG_DIR, args.model)
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    PRINT_AND_LOG('args:\n'+str(args))
    PRINT_AND_LOG('cfg:\n'+str(cfg))
    PRINT_AND_LOG('********** PASTA CONFIGS **********')
    PRINT_AND_LOG('pasta mode: %d' % args.pasta_mode)
    PRINT_AND_LOG('pasta trained: %s' % args.pasta_trained)

    Trainval_GT_path = osp.join(cfg.DATA_DIR, 'Trainval_GT_HAKE_Larger_93_verb_on_1_lmdb')
    Trainval_N_path = osp.join(cfg.DATA_DIR, 'Trainval_Neg_HAKE_Larger_93_lmdb')
    
    np.random.seed(cfg.RNG_SEED)

    # load data
    tb_dir       = cfg.ROOT_DIR + '/logs/' + args.model + '/'
    train_set    = hake_train(Trainval_GT_path, Trainval_N_path, args.Pos_augment, args.Neg_select)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True)
    data_length  = len(train_loader)
    PRINT_AND_LOG('Constructing training set ready, data length: {}'.format(data_length))

    # build network
    net   = pasta_res50(args)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.base_lr, momentum=0.9)
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2*data_length, T_mult=2, eta_min=0)
    elif args.lr_schedule != 'constant':
        raise NotImplementedError
    else:
        scheduler = None
    global_iter = 0

    # load weight if continue training
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        assert len(cfg.WEIGHT_PATH) > 0
        PRINT_AND_LOG("==> loading weight from: {}".format(cfg.WEIGHT_PATH))
        net, optimizer, global_iter = load_model(net, optimizer, scheduler, cfg.WEIGHT_PATH, args.load_optim)

    net       = net.cuda()
    train(net, train_loader, optimizer, scheduler, global_iter, tb_dir, cfg.TRAIN.MAX_EPOCH)