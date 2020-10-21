import torch
import os
import re
import time
import yaml
import numpy as np
import argparse
import pickle
import logging
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from prefetch_generator import BackgroundGenerator
from dataset import HICO_train_set, HICO_test_set
from model import AE, IDN
from utils import Timer, HO_weight, AverageMeter
from HICO_DET_utils import obj_range

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

models = {
    'AE': AE,
    'IDN': IDN
}

optims = {}
optims['SGD']     = optim.SGD

gpus = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
device = torch.device('cuda:{}'.format(gpus[0]))


def parse_arg():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--exp', dest='exp',
            help='Define exp name',
            default='_'.join(time.asctime(time.localtime(time.time())).split()), type=str)
    parser.add_argument('--config_path', dest='config_path',
            help='Select config file',
            default='configs/default.yml', type=str)
    args = parser.parse_args()
    return args

def get_config(args):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    config = edict(yaml.load(open(args.config_path, 'r'), Loader=loader))
    return config
    
def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    writer = SummaryWriter(os.path.join(cur_path, 'tb'))
    
    return logger, writer

args = parse_arg()

cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
assert not os.path.exists(cur_path), 'Duplicate exp name'
os.mkdir(cur_path)

config = get_config(args)
yaml.dump(dict(config), open(os.path.join(cur_path, 'config.yml'), 'w'))

logger, writer = get_logger(cur_path)
logger.info("Start print log")

train_set    = HICO_train_set(config, split='trainval', train_mode=True)
train_loader = DataLoaderX(train_set, batch_size=config.TRAIN.DATASET.BATCH_SIZE, shuffle=True,  num_workers=config.TRAIN.DATASET.NUM_WORKERS, collate_fn=train_set.collate_fn, pin_memory=False, drop_last=False)
logger.info("Train set loaded")

net = models[config.MODE](config.MODEL, HO_weight)
logger.info(net)
if len(gpus) > 1:
    net = torch.nn.DataParallel(net.to(device), device_ids=gpus, output_device=gpus[0])
else:
    net = net.to(device)

optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](net.parameters(), lr=config.TRAIN.OPTIMIZER.lr, momentum=config.TRAIN.OPTIMIZER.momentum, weight_decay=config.TRAIN.OPTIMIZER.weight_decay)
    
train_timer = Timer()

cur_epoch = 0
step = 0

def train(net, loader, optimizer, timer, epoch):
    net.train()
    global step
    
    timer.tic()
    meters = {
        'L_rec': AverageMeter(),
        'L_cls': AverageMeter(),
        'L_ae': AverageMeter(),
        'loss': AverageMeter()
    }
    for i, batch in enumerate(loader):
        
        n = batch['spatial'].shape[0]
        
        batch['spatial']    = batch['spatial'].cuda(non_blocking=True)
        batch['labels_s']   = batch['labels_s'].cuda(non_blocking=True)
        batch['labels_r']   = batch['labels_r'].cuda(non_blocking=True)
        batch['labels_ro']  = batch['labels_ro'].cuda(non_blocking=True)
        batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
        batch['sub_vec']    = batch['sub_vec'].cuda(non_blocking=True)
        batch['obj_vec']    = batch['obj_vec'].cuda(non_blocking=True)
        batch['uni_vec']    = batch['uni_vec'].cuda(non_blocking=True)
        
        output = net(batch)
        loss   = torch.mean(output['loss'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for key in output.keys():
            if key in meters:
                meters[key].update(torch.mean(output[key]).detach().cpu().data, n)

        timer.toc()
        timer.tic()
        if i % 400 == 0:
            for key in meters.keys():
                if key in output:
                    writer.add_scalar(key, torch.mean(output[key]).detach().cpu().data, step)
            if i % 2000 == 0:
                print("%03d epoch, %05d iter, average time %.4f, loss %.4f" % (epoch, i, timer.average_time, loss.detach().cpu().data))
        step += 1
    timer.toc()
    
    return net, meters

for i in range(config.TRAIN.MAX_EPOCH):
    train_str = "%03d epoch training" % i
    net, train_meters = train(net, train_loader, optimizer, train_timer, i)
    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    state = {'state':net.state_dict(), 'epoch': i}
    logger.info(train_str)
    torch.save(state, os.path.join(cur_path, 'epoch_%d.pth' % i))
