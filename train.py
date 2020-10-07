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
from HICO_DET_utils import obj_range, get_map

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
test_set    = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
test_loader = DataLoaderX(test_set, batch_size=config.TEST.BATCH_SIZE, shuffle=False, num_workers=config.TEST.NUM_WORKERS, collate_fn=test_set.collate_fn, pin_memory=False, drop_last=False)
logger.info("Test set loaded")


net = models[config.MODE](config.MODEL, HO_weight)
logger.info(net)
if len(gpus) > 1:
    net = torch.nn.DataParallel(net.to(device), device_ids=gpus, output_device=gpus[0])
else:
    net = net.to(device)

optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](net.parameters(), lr=config.TRAIN.OPTIMIZER.lr, momentum=config.TRAIN.OPTIMIZER.momentum, weight_decay=config.TRAIN.OPTIMIZER.weight_decay)
    
train_timer, test_timer = Timer(), Timer()

cur_epoch = 0
bst = 0.0
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

def eval(net, loader, timer, epoch):
    net.eval()

    verb_mapping = torch.from_numpy(pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')).float()

    bboxes, scores, keys = [], [], []
    for i in range(80):
        bboxes.append([])
        scores.append([])
        keys.append([])
    
    timer.tic()
    meters = {
        'L_rec': AverageMeter(),
        'L_cls': AverageMeter(),
        'L_ae': AverageMeter(),
        'loss': AverageMeter()
    }
    for i, batch in enumerate(loader):
        
        n = batch['shape'].shape[0]
        
        batch['shape']   = batch['shape'].cuda(non_blocking=True)
        batch['spatial'] = batch['spatial'].cuda(non_blocking=True)
        batch['sub_vec'] = batch['sub_vec'].cuda(non_blocking=True)
        batch['obj_vec'] = batch['obj_vec'].cuda(non_blocking=True)
        batch['uni_vec'] = batch['uni_vec'].cuda(non_blocking=True)
        batch['labels_s']   = batch['labels_s'].cuda(non_blocking=True)
        batch['labels_ro']  = batch['labels_ro'].cuda(non_blocking=True)
        batch['labels_r']  = batch['labels_r'].cuda(non_blocking=True)
        batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
        verb_mapping = verb_mapping.cuda(non_blocking=True)
        output = net(batch)
        
        for key in output.keys():
            if key in meters:
                meters[key].update(torch.mean(output[key]).detach().cpu().data, n)
        
        prob = torch.matmul(output['p'], verb_mapping)
        prob = prob.detach().cpu().numpy()
        batch['spatial'][:, 0] *= batch['shape'][:, 0]
        batch['spatial'][:, 1] *= batch['shape'][:, 1]
        batch['spatial'][:, 2] *= batch['shape'][:, 0]
        batch['spatial'][:, 3] *= batch['shape'][:, 1]
        batch['spatial'][:, 4] *= batch['shape'][:, 0]
        batch['spatial'][:, 5] *= batch['shape'][:, 1]
        batch['spatial'][:, 6] *= batch['shape'][:, 0]
        batch['spatial'][:, 7] *= batch['shape'][:, 1]
        obj_class = batch['obj_class']
        bbox = batch['spatial'].detach().cpu().numpy()
        hdet = batch['hdet']
        odet = batch['odet']

        for j in range(bbox.shape[0]):
            cls = obj_class[j]
            x, y = obj_range[cls][0]-1, obj_range[cls][1]
            keys[cls].append(batch['key'][j])
            bboxes[cls].append(bbox[j])
            scores[cls].append(prob[j, x:y] * hdet[j] * odet[j])
        timer.toc()
        timer.tic()
        if i % 3000 == 0:
            print("%03d epoch, %05d iteration, average time %.4f" % (epoch, i, timer.average_time))
    timer.toc()
    for i in range(80):
        keys[i]   = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])
    map, mrec = get_map(keys, scores, bboxes)
    return map, meters

for i in range(config.TRAIN.MAX_EPOCH):
    net, train_meters = train(net, train_loader, optimizer, train_timer, i)
    train_str = "%03d epoch training" % i

    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    logger.info(train_str)
    map, test_meters  = eval(net, test_loader, test_timer, i)
    
    ap = np.mean(map)
    test_str = "%03d epoch evaluation, mAP=%.2f" % (i, ap * 100)
    for (key, value) in test_meters.items():
        test_str += ", %s=%.4f" % (key, value.avg)
    logger.info(test_str)
    
    try:
        state = {
            'state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
            'config': net.config,
            'map': map,
            'step': step
        }
    except:
        state = {
            'state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
            'config': net.module.config,
            'map': map,
            'step': step
        }
    if ap > bst:
        bst = ap
        torch.save(state, os.path.join(cur_path, 'bst.pth'))
    torch.save(state, os.path.join(cur_path, 'latest.pth'))