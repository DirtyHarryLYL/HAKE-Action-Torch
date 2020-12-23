#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Dec 8th, 2020             #
#############################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import numpy as np
import argparse
from tqdm import tqdm
import pprint

from benchmark import benchmark
from test_net import custom_collate, test
from activity2vec.networks.pasta_net import pasta_res50
from activity2vec.dataset.hake_dataset import hake_train, hake_test
from activity2vec.ult.config import get_cfg
from activity2vec.ult.timer import Timer
from activity2vec.ult.loss import sigmoid_focal_loss
from anycurve import losscurve

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Training Activity2Vec')
    parser.add_argument('--model', dest='model',
            help='model name',
            default='', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-freeze', action='store_true')
    parser.add_argument('--checkpoint', default="", type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--module-trained', default='foot,leg,hip,hand,arm,head,verb', type=str)
    parser.add_argument('--ckpt-interval', default=cfg.TRAIN.CHECKPOINT_INTERVAL, type=int)
    parser.add_argument('--show-interval', default=1000, type=int)
    parser.add_argument('--lr', default=cfg.TRAIN.LEARNING_RATE, type=float)
    parser.add_argument('--lr-schedule', default=cfg.TRAIN.LR_SCHEDULE, type=str)
    parser.add_argument('--human-per-image', default=cfg.TRAIN.HUMAN_PER_IM, type=int)
    parser.add_argument('--pos-ratio', default=cfg.TRAIN.POS_RATIO, type=float)
    parser.add_argument('--fc-dim', default=cfg.MODEL.NUM_FC, type=int)
    parser.add_argument('--no-part-box', action='store_true')
    parser.add_argument('--no-loss-wts', action='store_true')
    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('--data-splits', default='', type=str)
    args = parser.parse_args()
    return args

def save_model(net, optimizer, scheduler, iters, ckp_path):
    torch.save({
                "iters": iters,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, ckp_path)

def load_model(cfg, model, optimizer, scheduler, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for state in scheduler.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if 'iters' in checkpoint:
        iters = checkpoint['iters']
    else:
        iters = 0

    return model, optimizer, scheduler, iters

def print_and_log(message, cfg):
    print(message)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    with open(os.path.join(cfg.LOG_DIR, "train.log"), 'a') as f:
        f.write(message + "\n")

def annos_to_cuda(cfg, annos):
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
    return annos

def train(cfg, net, train_loader, test_loader, optimizer, scheduler, global_iter=0):
    pasta_weights = np.load(cfg.DATA.PASTA_WEIGHTS_PATH, allow_pickle=True).item()
    for key in pasta_weights:
        pasta_weights[key] = torch.from_numpy(pasta_weights[key]).float().reshape(1, -1).cuda()
    
    data_length = len(train_loader)

    if cfg.TRAIN.SHOW_LOSS_CURVE:
        loss_curve_dir = os.path.join(cfg.LOG_DIR, 'loss_curve_db')
        
        loss_curve = losscurve(db_path=loss_curve_dir, db_name='data', figsize=(20, 12))
        loss_curve.add_key('loss')
        loss_curve.add_key('pasta_map')
        loss_curve.add_key('verb_map')
        
        loss_curve.set_xlabel('iteration')
        loss_curve.set_ylabel('loss', False)
        loss_curve.set_ylabel('mAP', True)
        loss_curve.daemon(True, cfg.TRAIN.SHOW_INTERVAL // cfg.TRAIN.DISPLAY_INTERVAL)

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
            annos = annos_to_cuda(cfg, annos)
            s_parts, s_verb = net(image, annos)
            loss = None
            
            for module_name in cfg.MODEL.MODULE_TRAINED:
                if module_name != 'verb':
                    pasta_label = annos['pasta'][module_name]
                    pasta_idx = net.pasta_name2idx[module_name]

                    if cfg.TRAIN.LOSS_TYPE == 'bce':
                        if cfg.TRAIN.WITH_LOSS_WTS:
                            bce_criterion = nn.BCEWithLogitsLoss(weight=pasta_weights[module_name].repeat(s_parts[pasta_idx].shape[0], 1), reduction='mean').cuda()
                        else:
                            bce_criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
                        if loss is None:
                            loss = bce_criterion(s_parts[pasta_idx], pasta_label)
                        else:
                            loss += bce_criterion(s_parts[pasta_idx], pasta_label)
                    elif cfg.TRAIN.LOSS_TYPE == 'focal':
                        if cfg.TRAIN.WITH_LOSS_WTS:
                            if loss is None:
                                loss = sigmoid_focal_loss(s_parts[pasta_idx], pasta_label, weight=pasta_weights[module_name].repeat(s_parts[pasta_idx].shape[0], 1), reduction='mean')
                            else:
                                loss += sigmoid_focal_loss(s_parts[pasta_idx], pasta_label, weight=pasta_weights[module_name].repeat(s_parts[pasta_idx].shape[0], 1), reduction='mean')
                        else:
                            if loss is None:
                                loss = sigmoid_focal_loss(s_parts[pasta_idx], pasta_label, weight=None, reduction='mean')
                            else:
                                loss += sigmoid_focal_loss(s_parts[pasta_idx], pasta_label, weight=None, reduction='mean')
                    else:
                        raise NotImplementedError
                else:
                    verb_label = annos['verbs']
                    if cfg.TRAIN.LOSS_TYPE == 'bce':
                        if cfg.TRAIN.WITH_LOSS_WTS:
                            bce_criterion = nn.BCEWithLogitsLoss(weight=pasta_weights[module_name].repeat(s_verb.shape[0], 1), reduction='mean').cuda()
                        else:
                            bce_criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
                        if loss is None:
                            loss = bce_criterion(s_verb, verb_label)
                        else:
                            loss += bce_criterion(s_verb, verb_label)
                    elif cfg.TRAIN.LOSS_TYPE == 'focal':
                        if cfg.TRAIN.WITH_LOSS_WTS:
                            if loss is None:
                                loss = sigmoid_focal_loss(s_verb, verb_label, weight=pasta_weights[module_name].repeat(s_verb.shape[0], 1), reduction='mean')
                            else:
                                loss += sigmoid_focal_loss(s_verb, verb_label, weight=pasta_weights[module_name].repeat(s_verb.shape[0], 1), reduction='mean')
                        else:
                            if loss is None:
                                loss += sigmoid_focal_loss(s_verb, verb_label, weight=None, reduction='mean')
                            else:
                                loss += sigmoid_focal_loss(s_verb, verb_label, weight=None, reduction='mean')
                    else:
                        raise NotImplementedError
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
                print_and_log("==> Saving weight: iteration {}".format(global_iter), cfg)
                weight_path = os.path.join(cfg.WEIGHT_DIR, "model_{}.pth".format(global_iter))
                save_model(net, optimizer, scheduler, global_iter, weight_path)
                
                if global_iter != 0:
                    # test in training
                    model_dir, model_filename = os.path.split(weight_path)
                    model_name = os.path.split(model_dir)[-1]
                    output_dir = os.path.join(cfg.ROOT_DIR, 'results', model_name, model_filename+'_results')

                    net.eval()
                    with torch.no_grad():
                        test(cfg, net, test_loader, output_dir)
                        pasta_map, verb_map = benchmark(output_dir, cfg)
                        loss_curve.log({'iteration': global_iter, 'pasta_map': pasta_map, 'verb_map': verb_map})
                    net.train()

            # show training schedule
            if global_iter % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                eta = int((len(train_loader) - (i + 1)) * global_timer.average_time)
                hours = eta // 3600
                minutes = (eta % 3600) // 60
                secs = (eta % 3600) % 60
                out_str = "|iteration: {:6d}|loss: {:.6f}|lr: {:.6f}|time: {:3f}|net_time: {:3f}|data_time: {:3f}|eta: {:2d}:{:2d}:{:2d}|".format(global_iter, mean_loss, float(optimizer.param_groups[0]['lr']), global_timer.average_time, net_timer.average_time, data_timer.average_time, hours, minutes, secs)
                print_and_log(out_str, cfg)

                if cfg.TRAIN.SHOW_LOSS_CURVE:
                    loss_curve.log({'iteration': global_iter, 'loss': mean_loss})
                    to_draw = loss_curve.daemon()
                    if to_draw:
                        loss_curve.clean()
                        loss_curve.draw('iteration', 'loss', cfg.MODEL_NAME + '_loss')
                        loss_curve.twin()

                        loss_curve.clean()
                        loss_curve.draw('iteration', 'pasta_map', cfg.MODEL_NAME + '_pasta')
                        loss_curve.draw('iteration', 'verb_map', cfg.MODEL_NAME + '_verb')
                        loss_curve.twin()

                        loss_curve.reset_choice()
                        loss_curve.legend(inside=False)
                        loss_curve.synchronize()
                        loss_curve.render(os.path.join(cfg.LOG_DIR, 'curve.png'))

            global_iter += 1

            global_timer.tic()
            data_timer.tic()

if __name__ == "__main__":
    # arg parsing
    cfg = get_cfg()
    args = parse_args(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    assert len(args.model) != 0
    assert cfg.TRAIN.IM_BATCH_SIZE == 1
    cfg.TRAIN.CHECKPOINT_INTERVAL    = args.ckpt_interval
    cfg.TRAIN.RESUME                 = args.resume
    cfg.TRAIN.CHECKPOINT_PATH        = args.checkpoint
    cfg.TRAIN.LEARNING_RATE          = args.lr
    cfg.TRAIN.LR_SCHEDULE            = args.lr_schedule
    cfg.TRAIN.HUMAN_PER_IM           = args.human_per_image
    cfg.TRAIN.POS_RATIO              = args.pos_ratio
    cfg.TRAIN.FREEZE_BACKBONE        = not args.no_freeze
    cfg.MODEL.NUM_FC                 = args.fc_dim
    if args.module_trained == 'all':
        cfg.MODEL.MODULE_TRAINED     = ['foot', 'leg', 'hip', 'hand', 'arm', 'head', 'verb']
    elif args.module_trained == 'all_pasta':
        cfg.MODEL.MODULE_TRAINED     = ['foot', 'leg', 'hip', 'hand', 'arm', 'head']
    elif args.module_trained == 'verb':
        cfg.MODEL.MODULE_TRAINED     = ['verb']
    else:
        cfg.MODEL.MODULE_TRAINED     = args.module_trained.split(',')
    cfg.WEIGHT_DIR                   = os.path.join(cfg.WEIGHT_DIR, args.model)
    cfg.LOG_DIR                      = os.path.join(cfg.LOG_DIR, args.model)
    cfg.MODEL_NAME                   = args.model
    cfg.TRAIN.SHOW_INTERVAL          = args.show_interval
    if len(args.data_splits) > 0:
        cfg.TRAIN.DATA_SPLITS            = [x.strip() for x in args.data_splits.split(',')]
    if args.no_part_box:
        cfg.MODEL.PART_ROI_ENABLE = False
    if args.no_loss_wts:
        cfg.TRAIN.WITH_LOSS_WTS = False
    if args.focal_loss:
        cfg.TRAIN.LOSS_TYPE = 'focal'
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    
    print_and_log('==> Training Activity2Vec <==', cfg)
    print_and_log('==> cfg:\n'+str(pprint.pformat(cfg, indent=2)), cfg)
    np.random.seed(cfg.RNG_SEED)

    train_set    = hake_train(cfg)
    test_set     = hake_test(cfg)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.TRAIN.IM_BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.IM_BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=1)

    data_length  = len(train_loader)
    print_and_log('==> Data length: {}'.format(data_length), cfg)

    # build network
    net   = pasta_res50(cfg)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN.LEARNING_RATE, momentum=0.9)
    if cfg.TRAIN.LR_SCHEDULE == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2*data_length, T_mult=2, eta_min=0)
    elif cfg.TRAIN.LR_SCHEDULE != 'constant':
        raise NotImplementedError
    else:
        scheduler = None
    global_iter = 0

    # load weight if continue training
    if cfg.TRAIN.RESUME:
        assert len(cfg.TRAIN.CHECKPOINT_PATH) > 0
        print_and_log("==> Loading weight from: {}".format(cfg.TRAIN.CHECKPOINT_PATH), cfg)
        net, optimizer, scheduler, global_iter = load_model(cfg, net, optimizer, scheduler, cfg.TRAIN.CHECKPOINT_PATH)

    net       = net.cuda()
    train(cfg, net, train_loader, test_loader, optimizer, scheduler, global_iter)