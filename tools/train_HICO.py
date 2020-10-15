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
import os
import sys
import inspect
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import hakeaction
from hakeaction.models.DJRN import DJRN_ResNet50
from hakeaction.datasets.HICO_dataset import HICO_Trainset, HICO_Trainset_3D
from hakeaction.configs.config_DJRN import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train on HICO')

    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1800000, type=int)

    parser.add_argument('--iter', dest='iter',
                        help='iter to load',
                        default=1800000, type=int)

    parser.add_argument('--model', dest='model',
            help='Select model',
            default='DJR_HICO', type=str)

    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one.',
            default=15, type=int)

    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)

    parser.add_argument('--train_continue', dest='train_continue',
            help='Whether to continue from previous weight',
            default=cfg.TRAIN_MODULE_CONTINUE, type=int)

    parser.add_argument('--train_module', dest='train_module',
            help='Training loss setting',
            default=cfg.TRAIN_MODULE, type=int)

    parser.add_argument('--weight',
            help='the path of weight to load from',
            default="",
            type=str)

    args = parser.parse_args()
    return args

def saveWeight(net, optimizer, epoch, loss, ckp_path):
    torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'loss': loss,
                }, ckp_path)

def load_model(model, ckp_path, optimizer=None):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, epoch, loss

    return model, epoch, loss

def PRINT_AND_LOG(message):
    print(message)
    with open(cfg.LOG_DIR + "HICO_train_log.txt", 'a') as f:
        f.write(message + "\n")

'''
    main training function for HICO-det dataset
'''
def train(net, train_loader, optimizer, lr_scheduler, tb_dir, max_epoch, cfg, startEpoch=0):
    latestSavedWeight = None
    for epoch in range(startEpoch, max_epoch):
        minLossPerEpoch = sys.maxsize
        lossPerEpoch    = 0.0

        for i, blobs in tqdm(enumerate(train_loader)):
            # forward inference
            flag = net(blobs)

            # failed in crop_and_pool
            if flag == False:
                print("iter:{}, fail in crop_and_pool".format(i))
                continue

            if   cfg.TRAIN_3D_BRANCH:
                total_loss = net.add_loss()
            elif cfg.TRAIN_2D_BRANCH:
                total_loss = net.add_loss_2D_branch()
            elif cfg.TRAIN_BSH_BRANCH:
                total_loss = net.add_loss_BSH_branch()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lossPerEpoch += float(total_loss)

            if i % cfg.TRAIN.DISPLAY == 0:
                PRINT_AND_LOG("Epoch {}, iteration {}, lossPerBatch={:.6f}".format(epoch, i, float(lossPerEpoch / (i+1))))

        # log per epoch
        PRINT_AND_LOG("Epoch {}, lossPerEpoch={:.6f}".format(epoch, lossPerEpoch))

        # save best model
        if lossPerEpoch < minLossPerEpoch:
            PRINT_AND_LOG("==> saving weight: Epoch {}, lossPerEpoch={:.6f}".format(epoch, lossPerEpoch))
            minLossPerEpoch = lossPerEpoch

            if cfg.ONLY_KEEP_BEST_WEIGHT and latestSavedWeight != None:
                os.remove(latestSavedWeight)
        
            latestSavedWeight = cfg.WEIGHT_DIR + "HICO_DJRN_epoch_{}_loss_{}.tar".format(epoch, lossPerEpoch)
            saveWeight(net, optimizer, epoch, lossPerEpoch, latestSavedWeight)

    # save model of last epoch
    latestSavedWeight = cfg.WEIGHT_DIR + "HICO_weight_lastEpoch_epoch_{}.tar".format(epoch)
    saveWeight(net, optimizer, epoch, lossPerEpoch, latestSavedWeight)

if __name__ == "__main__":
	# TODO: select network, dataset via args

    # arg parsing
    args = parse_args()
    cfg.TRAIN_MODULE_CONTINUE   = args.train_continue
    cfg.WEIGHT_PATH             = args.weight
    cfg.TRAIN_MODULE            = args.train_module
    np.random.seed(cfg.RNG_SEED)

    # data preparing
    ''' # 2D dataset
    Trainval_GT  = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl',  "rb" ), encoding="bytes")
    Trainval_N   = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb" ), encoding="bytes")
    '''
    Trainval_GT  = pickle.load( open( '/Disk3/zhanke/data/pami/Trainval_GT_HICO_with_smplx.pkl',  "rb" ), encoding="bytes")
    Trainval_N   = pickle.load( open( '/Disk3/zhanke/data/pami/Trainval_Neg_HICO_with_smplx.pkl', "rb" ), encoding="bytes")
    # Trainval_GT  = pickle.load( open( '/Disk3/zhanke/data/pami/Trainval_GT_HICO_with_smplx_demo.pkl',  "rb" ), encoding="bytes")
    # Trainval_N   = pickle.load( open( '/Disk3/zhanke/data/pami/Trainval_Neg_HICO_with_smplx_demo.pkl', "rb" ), encoding="bytes")

    tb_dir       = cfg.LOG_DIR
    # train_set    = HICO_Trainset(Trainval_GT, Trainval_N, cfg, Pos_augment=args.Pos_augment, Neg_select=args.Neg_select)  # 2D dataset 
    train_set    = HICO_Trainset_3D(Trainval_GT, Trainval_N, cfg, Pos_augment=args.Pos_augment, Neg_select=args.Neg_select) # 3D dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=False)

    # build network & training settings
    net          = DJRN_ResNet50()
    optimizer    = optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*len(Trainval_GT))
    epoch        = 1

    # load weight if continue training
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        PRINT_AND_LOG("==> loading weight from: {}".format(cfg.WEIGHT_PATH))
        net, epoch, loss = load_model(net, cfg.WEIGHT_PATH)
        epoch += 1

    net.freezeParts(cfg.PARTS_TO_FREEZE)
    net = net.cuda()
    train(net, train_loader, optimizer, lr_scheduler, tb_dir, cfg.TRAIN.MAX_EPOCH, cfg, startEpoch=epoch)
