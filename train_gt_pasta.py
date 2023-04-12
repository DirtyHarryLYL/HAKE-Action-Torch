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

from model import NLR_simplified_no_T
from util.utils import Timer, HO_weight, AverageMeter
from util.HICO_DET_utils import obj_range, get_map, getSigmoid, get_map_with_ko
from dataset import (
    HICO_test_set,
    HICO_train_set,
    HICO_gt_test_set,
    qpic_test_set,
)
from util.object_to_hoi import obj_range_padding, obj_range_cnt, obj_range_extra
from util.pasta_utils import pvp_weight


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


optims = {}
optims["RMSprop"] = optim.RMSprop
optims["SGD"] = optim.SGD
optims["AdamW"] = optim.AdamW

gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
device = torch.device("cuda:{}".format(gpus[0]))

models = {"NLR_simplified_no_T": NLR_simplified_no_T}
dataset_mapping={
    "HICO_train_set": HICO_train_set,
    "HICO_test_set": HICO_test_set,
    "HICO_gt_test_set": HICO_gt_test_set,
    "qpic_test_set": qpic_test_set,
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
    parser.add_argument(
        "--human_level",
        dest="human_level",
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
if not os.path.exists(cur_path):
    os.mkdir(cur_path)
# assert not os.path.exists(cur_path), "Duplicate exp name"
# os.mkdir(cur_path)

config = get_config(args)
config, ori_ckpt = get_resume(config)
yaml.dump(dict(config), open(os.path.join(cur_path, "config.yml"), "w"))

if config.MODE != "Linear":
    set_seed(config.RANDOM_SEED)

logger, writer = get_logger(cur_path)
logger.info("Start print log")

if not args.eval and not args.eval_for_store:
    train_set = HICO_train_set(config)

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

test_set = dataset_mapping[config.TEST.DATA_SET](config)

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
logger.info(net)
# if ori_ckpt or args.eval:
if ori_ckpt:
    pretrained_dict = ori_ckpt['state'].state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print("state loaded.")

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

cur_epoch = 0
bst = 0.0
step = 0
last_epoch = -1

if ori_ckpt and config.TRAIN.OPTIMIZER.RESUME:
    optimizer.load_state_dict(ori_ckpt["optim_state"])
    last_epoch = ori_ckpt["step"]
    step = ori_ckpt["step"]
if "LR_SCHEDULER" in config.TRAIN:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.TRAIN.LR_SCHEDULER.T_MAX,
        last_epoch=last_epoch,
    )

train_timer, test_timer = Timer(), Timer()

def train(net, loader, optimizer, timer, epoch):
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

        try:
            n = batch["gt_label"].shape[0]
        except:
            n = batch["labels_r"].shape[0]

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
            'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r',
            'rule_lens', 'rule_cnt'
        ]:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

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

def eval(net, loader, timer, epoch, eval=False, eval_ko = False):
    net.eval()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(False)

    verb_mapping = torch.from_numpy(
        pickle.load(open("util/verb_mapping.pkl", "rb"), encoding="latin1")
    ).float()

    bboxes, scores, keys = [], [], []
    for i in range(80):
        bboxes.append([])
        scores.append([])
        keys.append([])

    if eval:
        hdets, odets, scores_wo_lis = [], [], []
        for i in range(80):
            hdets.append([])
            odets.append([])
            scores_wo_lis.append([])

    timer.tic()
    # meters = {
    #     'loss': AverageMeter()
    # }

    for i, batch in enumerate(loader):

        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part', 'rule_cnt',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)
        verb_mapping = verb_mapping.cuda(non_blocking=True)

        output = net(batch)

        bbox = batch["spatial"]

        obj_class = batch["obj_class"]
        hdet = batch["hdet"]
        odet = batch["odet"]

        prob = output["p"].detach().cpu().numpy()
        sc = output["s"].detach().cpu().numpy()

        for j in range(bbox.shape[0]):
            cls = obj_class[j]
            x, y = obj_range[cls][0] - 1, obj_range[cls][1]
            keys[cls].append(batch["key"][j])
            bboxes[cls].append(bbox[j])

            if cls in obj_range_extra:
                p = prob[j, -obj_range_cnt[cls]:]
                s = sc[j, -obj_range_cnt[cls]:]
            else:
                p = prob[j, : obj_range_cnt[cls]]
                s = sc[j, : obj_range_cnt[cls]]

            if not config.TEST.get('LIS') or config.TEST.LIS:
                scores[cls].append(
                    p * getSigmoid(9, 1, 3, 0, hdet[j]) * getSigmoid(9, 1, 3, 0, odet[j])
                )
            else:
                scores[cls].append(
                    p * hdet[j] * odet[j]
                )

            if eval:
                hdets[cls].append(hdet[j])
                odets[cls].append(odet[j])
                scores_wo_lis[cls].append(s)

        timer.toc()
        timer.tic()
        if i % 100 == 0:
            print(
                "%03d epoch, %05d iteration, average time %.4f"
                % (epoch, i, timer.average_time)
            )

        # if (i + 1) % 10 == 0:
        # logic_embed = [np.concatenate(_) for _ in zip(*logic_embed)]
        # with open(os.path.join(cur_path, 'logic_embed.pkl'), 'wb') as f:
        # pickle.dump(logic_embed, f)
        # assert False

    timer.toc()

    for i in range(80):
        keys[i] = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])

        if eval:
            hdets[i] = np.array(hdets[i])
            odets[i] = np.array(odets[i])
            scores_wo_lis[i] = np.array(scores_wo_lis[i])

    if eval_ko:
        map, _, map_ko, _ = get_map_with_ko(keys, scores, bboxes)
        return map, map_ko

    map, mrec = get_map(keys, scores, bboxes)

    if eval:
        with open(os.path.join(cur_path, 'result.pkl'), 'wb') as f:
            pickle.dump({'keys': keys, 'scores': scores_wo_lis, 'bboxes': bboxes, 'hdet': hdets, 'odet': odets}, f)

    return map

if args.eval:
    map = eval(net, test_loader, test_timer, 0, eval=True)
    print(np.mean(map))
    with open(os.path.join(cur_path, 'map.pkl'), 'wb') as f:
        pickle.dump(map, f)
    quit()


# export CUDA_VISIBLE_DEVICES=7; python train_gt_pasta.py --config_path configs/gt-pasta_gt-bbox.yml --exp gt-pasta_gt-bbox_eval --eval

bst = 0.0

for i in range(config.TRAIN.MAX_EPOCH):
    # if i > 0 or not ori_ckpt:
    net, train_meters = train(net, train_loader, optimizer, train_timer, i)
    if "LR_SCHEDULER" in config.TRAIN:
        scheduler.step()
    train_str = "%03d epoch training" % i

    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    logger.info(train_str)

    map = eval(net, test_loader, test_timer, i)
    ap = np.mean(map)
    test_str = "%03d epoch evaluation, mAP=%.2f" % (i, ap * 100)
    logger.info(test_str)

    try:
        state = {
            # 'state': net.state_dict(),
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.config,
            "map": map,
            "step": step,
            # "lr_scheduler_state": scheduler.state_dict(),
        }
    except:
        state = {
            # 'state': net.state_dict(),
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.module.config,
            "map": map,
            "step": step,
            # "lr_scheduler_state": scheduler.state_dict(),
        }
    if "LR_SCHEDULER" in config.TRAIN:
        state["lr_scheduler_state"] = scheduler.state_dict()
    if ap > bst:
        bst = ap
        torch.save(state, os.path.join(cur_path, "bst.pth"))
    torch.save(state, os.path.join(cur_path, "latest.pth"))
