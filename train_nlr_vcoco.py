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
import random

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator

from model import (
    NLR, Linear, NLR_simplified, NLR_simplified_no_T,
    NLR_10v, VL_align, Linear_10v, part_classifier,
    NLR_10v_simplified_no_T
)
from utils import Timer, HO_weight, AverageMeter
from HICO_DET_utils import getSigmoid
from from_1800.code.dataset import (
    VCOCO_train_set,
    VCOCO_test_set
)
# from dataset_nlr_10v import HICO_train_set_10v, HICO_test_set_10v
from pasta_utils import pvp_weight
from VCOCO_utils import get_map_vcoco_no_save, VERB2ID, ID2VERB, mapping


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


optims = {}
optims["RMSprop"] = optim.RMSprop
optims["SGD"] = optim.SGD
optims["AdamW"] = optim.AdamW

# export CUDA_VISIBLE_DEVICES=2; python train_nlr.py --config_path configs/default_nlr_pvp.yml
# watch -n 1 'grep mAP exp/Mon_Jul_6_13:37:22_2020/log.txt | tail'

gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
device = torch.device("cuda:{}".format(gpus[0]))

models = {"NLR": NLR,
          "Linear": Linear,
          "NLR_simplified": NLR_simplified,
          "NLR_10v": NLR_10v,
          "NLR_simplified_no_T": NLR_simplified_no_T,
          "NLR_10v_simplified_no_T": NLR_10v_simplified_no_T,
          "VL_align": VL_align,
          "Linear_10v": Linear_10v,
          }


def parse_arg():
    parser = argparse.ArgumentParser(description="Generate detection file")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="Define exp name",
        default="_".join(time.asctime(time.localtime(time.time())).split()),
        type=str,
    )
    parser.add_argument(
        "--config_path",
        dest="config_path",
        help="Select config file",
        default="configs/default_nlr.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        help="Whether to load model from previous checkpoint",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--resume_optim",
        dest="resume_optim",
        help="Whether to load optimizer from previous checkpoint",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="The path of the checkpoint",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--max_epoch",
        dest="max_epoch",
        help="Max epoch to train",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        help="Whether activate evaluation only mode",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def get_resume(config):
    if config.RESUME:
        assert config.CHECKPOINT, "No checkpoint specified"
        ckpt = torch.load(config.CHECKPOINT)
        config["MODEL"].update(ckpt["config"], encoding="gdk")

    else:
        ckpt = None
    return config, ckpt


def get_config(args):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(
            u"""^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list(u"-+0123456789."),
    )

    config = edict(yaml.load(open(args.config_path, "r"), Loader=loader))
    return config


def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(os.path.join(cur_path, "tb"))

    return logger, writer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parse_arg()

cur_path = os.path.join(os.getcwd(), "exp", args.exp)
assert not os.path.exists(cur_path), "Duplicate exp name"
os.mkdir(cur_path)

config = get_config(args)
config, ori_ckpt = get_resume(config)
yaml.dump(dict(config), open(os.path.join(cur_path, "config.yml"), "w"))

if config.MODE != "Linear":
    set_seed(config.RANDOM_SEED)

logger, writer = get_logger(cur_path)
logger.info("Start print log")

if not args.eval:
    train_set = VCOCO_train_set(config, split='trainval', train_mode=True)

    train_loader = DataLoaderX(
        train_set,
        batch_size=config.TRAIN.DATASET.BATCH_SIZE,
        num_workers=config.TRAIN.DATASET.NUM_WORKERS,
        collate_fn=train_set.collate_fn,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
    )
    logger.info("Train set loaded")

test_set = VCOCO_test_set(config, split='test')

test_loader = DataLoaderX(
    test_set,
    batch_size=config.TEST.BATCH_SIZE,
    shuffle=False,
    num_workers=config.TEST.NUM_WORKERS,
    collate_fn=test_set.collate_fn,
    pin_memory=False,
    drop_last=False,
)
logger.info("Test set loaded")

net = models[config.MODE](config)
part_classifier = part_classifier(config.MODEL.PART_CLS, pvp_weight)

if len(gpus) > 1:
    part_classifier = torch.nn.DataParallel(part_classifier.to(device), device_ids=gpus, output_device=gpus[0])
else:
    part_classifier = part_classifier.to(device)

part_classifier.eval()

logger.info(net)
if ori_ckpt or args.eval:
    # net.load_state_dict(ori_ckpt['state'])
    # ori_stat = ori_ckpt['state']


    pretrained_dict = ori_ckpt['state'].state_dict()
    # net.load_state_dict(pretrained_dict)

    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print("state loaded.")
    # if config.MODE == 'NLR_simplified':
    #     net.load_state_dict(ori_ckpt['state'].state_dict())
    #     print("state loaded.")
    # else:
    #     net = ori_ckpt["state"]
    #     print("net loaded.")

    # if config.get('FROZEN', 'True'):
    #     for k, v in net.named_parameters():
    #         if 'pvp_embedding' in k:
    #             v.requires_grad = True
    #         else:
    #             v.requires_grad = False
    #
    #     for k, v in net.named_parameters():
    #         print(k, v.requires_grad)

    # if config.get('NO_LOGIC', 'True'):
    #     config.MODEL.LOSS.LOGIC_FAC = 0.0
    #     print('logic_fac = 0.0')

if len(gpus) > 1:
    net = torch.nn.DataParallel(net.to(device), device_ids=gpus, output_device=gpus[0])
else:
    net = net.to(device)

if config.TRAIN.OPTIMIZER.TYPE != 'AdamW':
    optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](
        net.parameters(),
        lr=config.TRAIN.OPTIMIZER.lr,
        momentum=config.TRAIN.OPTIMIZER.momentum,
        weight_decay=config.TRAIN.OPTIMIZER.weight_decay,
    )
else:
    optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](
        net.parameters(),
        lr=config.TRAIN.OPTIMIZER.lr,
        weight_decay=config.TRAIN.OPTIMIZER.weight_decay,
        amsgrad=True,
    )

if len(gpus) > 1:
    optimizer = torch.nn.DataParallel(optimizer, device_ids=gpus, output_device=gpus[0])
if ori_ckpt and config.TRAIN.OPTIMIZER.RESUME:
    optimizer.load_state_dict(ori_ckpt["optim_state"])

train_timer, test_timer = Timer(), Timer()

cur_epoch = 0
bst = 0.0
step = 0


def train(net, part_classifier, loader, optimizer, timer, epoch):
    net.train()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    global step

    timer.tic()
    meters = {
        "L_cls": AverageMeter(),
        "L_logic": AverageMeter(),
        "L_align": AverageMeter(),
        "L_action": AverageMeter(),
        "loss": AverageMeter(),
    }
    for i, batch in enumerate(loader):

        n = batch["labels_v"].shape[0]

        for key in [
            "gt_label",
            "gt_pvp",
            "gt_obj",
            "rule",
            "gt_range",
            "shuffle_index",
            'gt_part',
            'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5',
            'FO', 'FH', 'FR', 'FS',
            'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
            'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_v',
            'rule_lens'
        ]:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        batch["gt_pvp"] = torch.sigmoid(part_classifier(batch)['s'])

        output = net(batch)
        loss = torch.mean(output["loss"])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()

        for key in output.keys():
            if key in meters:
                meters[key].update(torch.mean(output[key]).detach().cpu().data, n)

        timer.toc()
        timer.tic()

        if i % 50 == 0:
            for key in meters.keys():
                if key in output:
                    writer.add_scalar(
                        key, torch.mean(output[key]).detach().cpu().data, step
                    )
            if i % 100 == 0:
                print(
                    "%03d epoch, %05d iter, average time %.4f, loss %.4f"
                    % (epoch, i, timer.average_time, loss.detach().cpu().data)
                )
        step += 1
    timer.toc()

    return net, meters


def eval(net, part_classifier, loader, timer, epoch, eval=False):
    net.eval()

    bboxes, scores, keys, classes, sub_id, logic_embed = [], [], [], [], [], []

    timer.tic()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(False)

    meters = {
        'L_part': AverageMeter(),
        'L_cls': AverageMeter(),
        'L_sAP': AverageMeter(),
        'loss': AverageMeter(),
    }

    for i, batch in enumerate(loader):

        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_v']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        batch["gt_pvp"] = torch.sigmoid(part_classifier(batch)['s'])

        output = net(batch)

        for key in output.keys():
            if key in meters:
                meters[key].update(torch.mean(output[key]).detach().cpu().data, n)

        prob = output["p"].detach().cpu().numpy()

        # batch['spatial'][:, [0, 2, 4, 6]] *= batch['shape'][:, 0:1]
        # batch['spatial'][:, [1, 3, 5, 7]] *= batch['shape'][:, 1:2]
        obj_class = batch['obj_class']
        bbox = batch['spatial']
        hdet = batch['hdet'].reshape(-1, 1)
        odet = batch['odet'].reshape(-1, 1)

        keys.append(batch['key'])
        bboxes.append(bbox)
        classes.append(obj_class)
        sub_id.append(batch['sub_id'])
        logic_embed.append(output["logic_embed"])
        if config.TEST.LIS:
            scores.append(prob * getSigmoid(6, 6, 7, 0, hdet) * getSigmoid(6, 6, 7, 0, odet))
        else:
            scores.append(prob * hdet * odet)
        timer.toc()
        timer.tic()
        if i % 100 == 0:
            print("%03d epoch, %05d iteration, average time %.4f" % (epoch, i, timer.average_time))
        if (i + 1) % 100 == 0:
            pickle.dump(logic_embed, open(os.path.join(cur_path, "logic_embed.pkl"), "wb"))
            assert False
    timer.toc()

    keys = np.concatenate(keys)
    bboxes = np.concatenate(bboxes)
    classes = np.concatenate(classes)
    sub_id = np.concatenate(sub_id)
    scores = np.concatenate(scores)
    map = get_map_vcoco_no_save(keys, sub_id, scores, bboxes, classes)
    reindex = []
    for i in range(29):
        verb = ID2VERB[i]
        if verb in ['point', 'stand', 'walk', 'run', 'smile']:
            reindex.append(np.nan)
            continue
        elif verb in mapping:
            x, y = mapping[verb]
        else:
            for key in mapping.keys():
                if verb in key:
                    x, y = mapping[key]
        reindex.append(map[x, y])

    return np.array(reindex)


if args.eval:
    map = eval(net, part_classifier, test_loader, test_timer, 0, eval=True)
    print(np.nanmean(map))
    quit()

bst = 0.0

if config.MODEL.CHECKPOINT:
    map = eval(net, part_classifier, test_loader, test_timer, -1)
    ap = np.nanmean(map)
    test_str = "-1 epoch evaluation, mAP=%.2f" % (ap * 100)
    logger.info(test_str)

if config.MODE == 'NLR_10v' and config.MODEL.NLR.CHECKPOINT:
    map = eval(net, part_classifier, test_loader, test_timer, -1)
    ap = np.nanmean(map)
    test_str = "-1 epoch evaluation, mAP=%.2f" % (ap * 100)
    logger.info(test_str)

if config.TEST.BASELINE:
    try:
        baseline = pickle.load(open(config.TEST.BASELINE, "rb"))
    except:
        baseline = pickle.load(open(config.TEST.BASELINE, "rb"), encoding="latin1")
    if isinstance(baseline, dict):
        baseline = np.array(baseline["map"])
    else:
        baseline = np.array(baseline)

if "LR_SCHEDULER" in config.TRAIN:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.TRAIN.LR_SCHEDULER.T_MAX
    )

for i in range(config.TRAIN.MAX_EPOCH):
    # if i > 0 or not ori_ckpt:
    net, train_meters = train(net, part_classifier, train_loader, optimizer, train_timer, i)
    if "LR_SCHEDULER" in config.TRAIN:
        scheduler.step()
    train_str = "%03d epoch training" % i

    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    logger.info(train_str)

    if (i + 1 >= config.TRAIN.get("EVAL_START_EPOCH", -1) \
        or i == 0 \
        or (i + 1) % config.TRAIN.get("EVAL_INTER", 1) == 0):
        map = eval(net, part_classifier, test_loader, test_timer, i)
        ap = np.nanmean(map)
        test_str = "%03d epoch evaluation, mAP=%.2f" % (i, ap * 100)
        logger.info(test_str)

    if config.TEST.BASELINE:
        delta = map - baseline
        delta = np.round(100 * delta, 2)
        index = np.where(delta > 0)[0]
        delta = delta[delta > 0]
        inf = dict(zip(index, delta))
        logger.info(str(inf))

    try:
        state = {
            # 'state': net.state_dict(),
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.config,
            "map": map,
            "step": step,
        }
    except:
        state = {
            # 'state': net.state_dict(),
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.module.config,
            "map": map,
            "step": step,
        }
    if ap > bst:
        bst = ap
        torch.save(state, os.path.join(cur_path, "bst.pth"))
    torch.save(state, os.path.join(cur_path, "latest.pth"))
