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
from tqdm import tqdm
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from lib.networks.TIN_HICO import TIN_ResNet50
from lib.dataset.HICO_dataset import HICO_Trainset
from lib.ult.config_TIN import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train PVP on HICO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1800000, type=int)
    parser.add_argument('--iter', dest='iter', 
                        help='iter to load', 
                        default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_HICO', type=str)
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

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss  = checkpoint["loss"]

    return model, epoch, loss

def PRINT_AND_LOG(message):
    print(message)
    with open(cfg.LOG_DIR + "HICO_train_log.txt", 'a') as f:
        f.write(message + "\n")

'''
    main training function for HICO_det dataset
'''
def train(net, train_loader, optimizer, tb_dir, max_epoch, startEpoch=0):
    latestSavedWeight = None

    for epoch in range(startEpoch+1, max_epoch):
        minLossPerEpoch = sys.maxsize
        lossPerEpoch    = 0.0

        for i, blobs in tqdm(enumerate(train_loader)):

            # forward inference
            flag = net(blobs)

            # failed in crop_and_pool
            if flag == False:
                print("iter:{}, fail in crop_and_pool".format(i))
                continue

            total_loss = net.add_loss()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lossPerEpoch += float(total_loss)

            if i % cfg.TRAIN.DISPLAY == 0:
                PRINT_AND_LOG("Epoch {}, iteration {}, lossPerBatch={:.6f}".format(epoch, i, float(lossPerEpoch / (i+1))))
        
        # log per epoch
        PRINT_AND_LOG("Epoch {}, lossPerEpoch={:.6f}".format(epoch, lossPerEpoch))
        
        # save best model
        if lossPerEpoch < minLossPerEpoch:
            PRINT_AND_LOG("==> saving weight: Epoch {}, lossPerEpoch={:.6f}".format(epoch, lossPerEpoch))
            minLossPerEpoch = lossPerEpoch

            '''
            if latestSavedWeight != None:
                os.remove(latestSavedWeight)
            '''
             
            latestSavedWeight = cfg.WEIGHT_DIR + "HICO_weight_epoch_{}_loss_{}.tar".format(epoch, lossPerEpoch)
            saveWeight(net, optimizer, epoch, lossPerEpoch, latestSavedWeight)

    # save model of last epoch
    latestSavedWeight = cfg.WEIGHT_DIR + "HICO_weight_lastEpoch_epoch_{}.tar".format(epoch)
    saveWeight(net, optimizer, epoch, lossPerEpoch, latestSavedWeight)

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    cfg.TRAIN_MODULE_CONTINUE   = args.train_continue
    cfg.TRAIN_MODULE            = args.train_module
    cfg.WEIGHT_PATH             = args.weight
    np.random.seed(cfg.RNG_SEED)

    # data preparing
    Trainval_GT  = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl',  "rb" ), encoding="bytes")
    Trainval_N   = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb" ), encoding="bytes")
    tb_dir       = cfg.ROOT_DIR + '/logs/' + args.model + '/'
    train_set    = HICO_Trainset(Trainval_GT, Trainval_N, args.Pos_augment, args.Neg_select)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True)
    
    # build network
    net   = TIN_ResNet50()
    epoch = 0

    # load weight if continue training
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        PRINT_AND_LOG("==> loading weight from: {}".format(cfg.WEIGHT_PATH))
        net, epoch, loss = load_model(net, cfg.WEIGHT_PATH)

    net       = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=0.9)
    train(net, train_loader, optimizer, tb_dir, cfg.TRAIN.MAX_EPOCH, startEpoch=epoch)
