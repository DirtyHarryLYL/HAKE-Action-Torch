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

from model import NLR, Linear, NLR_simplified, NLR_simplified_no_T, NLR_10v, VL_align, Linear_10v
from utils import Timer, HO_weight, AverageMeter
from HICO_DET_utils import obj_range, get_map, getSigmoid

from dataset_nlr_10v import HICO_train_set_10v, HICO_test_set_10v
from object_to_hoi import obj_range_padding, obj_range_cnt, obj_range_extra
from pasta_utils import pvp_weight
from cal_train_map import get_train_map


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


optims = {}
optims["RMSprop"] = optim.RMSprop
optims["SGD"] = optim.SGD
optims["AdamW"] = optim.AdamW

# export CUDA_VISIBLE_DEVICES=2; python train_nlr_update.py --config_path configs/default_nlr_gt_update.yml --exp rule_update
# watch -n 1 'grep mAP exp/rule_update_judger_avg_train_acc_new/log.txt | tail'

gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
device = torch.device("cuda:{}".format(gpus[0]))

models = {"NLR": NLR,
          "Linear": Linear,
          "NLR_simplified": NLR_simplified,
          "NLR_10v": NLR_10v,
          "NLR_simplified_no_T": NLR_simplified_no_T,
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
    parser.add_argument(
        "--eval_train_map",
        dest="eval_train_map",
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

if not config.get('PVP', False):
    from dataset_nlr import (
        HICO_test_set,
        HICO_train_set,
        HICO_test_set_without_rule,
        HICO_train_set_without_rule,
        HICO_gt_test_set,
    )
else:
    from dataset import HICO_test_set, HICO_train_set

if not args.eval:
    if config.MODE == "Linear":
        train_set = HICO_train_set_without_rule(config)
    elif config.MODE in ['NLR_10v', "VL_align", "Linear_10v"]:
        train_set = HICO_train_set_10v(config)
    else:
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

    update_loader = DataLoaderX(
        train_set,
        batch_size=int(config.TRAIN.DATASET.BATCH_SIZE / 4),
        num_workers=config.TRAIN.DATASET.NUM_WORKERS,
        collate_fn=train_set.collate_fn,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
    )

if config.MODE == "Linear":
    test_set = HICO_test_set_without_rule(config)
elif config.MODE in ['NLR_10v', "VL_align", "Linear_10v"]:
    test_set = HICO_test_set_10v(config)
elif config.get("GT_TEST", False):
    test_set = HICO_gt_test_set(config)
else:
    test_set = HICO_test_set(config)

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

train_timer, test_timer, eval_timer = Timer(), Timer(), Timer()

rule_all = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # (600,88,k)
select = None
rule_all_cnt = config.MODEL.NUM_RULE_ALL

if not config.MODEL.get('DYNAMIC', False):
    rule_select = pickle.load(open(config.MODEL.INITIAL_RULE, 'rb'))
    rule_select_cnt = config.MODEL.NUM_RULE
else:
    rule_select_ori = pickle.load(open(config.MODEL.INITIAL_RULE, 'rb'))['rules']  # (600,10,22)
    rule_select_cnt = config.MODEL.NUM_RULE
    rule_select = np.zeros([rule_select_ori.shape[0], rule_select_cnt, rule_select_ori.shape[2]])  # (600,12,22)
    rule_select[:, :rule_select_ori.shape[1], :] = rule_select_ori
    rule_adaptive = pickle.load(open(config.MODEL.INITIAL_RULE, 'rb'))['rules_cnt']  # (600)


def train(net, loader, optimizer, timer, epoch, rule_select):
    net.config.NUM_RULE = rule_select_cnt
    net.config.UPDATE = False

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

        # batch["rule"] (bz, 18, 10, k), batch["gt_range"] (bz, 18), rule (600, 10, k)
        rule = rule_select[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()
        if config.MODEL.get('DYNAMIC', False):
            batch["rule_cnt"] = torch.from_numpy(rule_adaptive[batch["gt_range"]]).long()  # (600),(bz,18)->(bz,18)
        for key in ["gt_label", "gt_pvp", "gt_obj", "rule", "gt_range", "shuffle_index", 'gt_part', "rule_cnt",
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5',
                    'FO', 'FH', 'FR', 'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5',
                    'labels_r', 'rule_lens']:
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

        # break

    timer.toc()

    return net, meters


def eval(net, loader, timer, epoch, rule_select):
    net.config.NUM_RULE = rule_select_cnt
    net.config.UPDATE = False

    net.eval()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(False)

    bboxes, scores, keys = [], [], []
    for i in range(80):
        bboxes.append([])
        scores.append([])
        keys.append([])

    timer.tic()

    for i, batch in enumerate(loader):

        rule = rule_select[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()
        if config.MODEL.get('DYNAMIC', False):
            batch["rule_cnt"] = torch.from_numpy(rule_adaptive[batch["gt_range"]]).long()  # (600),(bz,18)->(bz,18)

        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part', "rule_cnt",
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)
        bbox = batch["spatial"]
        obj_class = batch["obj_class"]
        hdet = batch["hdet"]
        odet = batch["odet"]
        prob = output["p"].detach().cpu().numpy()

        for j in range(bbox.shape[0]):
            cls = obj_class[j]
            x, y = obj_range[cls][0] - 1, obj_range[cls][1]
            keys[cls].append(batch["key"][j])
            bboxes[cls].append(bbox[j])

            if cls in obj_range_extra:
                p = prob[j, -obj_range_cnt[cls]:]
            else:
                p = prob[j, : obj_range_cnt[cls]]

            if not config.TEST.get('LIS') or config.TEST.LIS:
                scores[cls].append(
                    p * getSigmoid(9, 1, 3, 0, hdet[j]) * getSigmoid(9, 1, 3, 0, odet[j])
                )
            else:
                scores[cls].append(
                    p * hdet[j] * odet[j]
                )

        timer.toc()
        timer.tic()
        if i % 100 == 0:
            print(
                "%03d epoch, %05d iteration, average time %.4f"
                % (epoch, i, timer.average_time)
            )

    timer.toc()

    for i in range(80):
        keys[i] = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])

    map, mrec = get_map(keys, scores, bboxes)

    return map


def eval_with_separate_rule_by_train_loss(net, loader, timer, epoch, rule_set, rule_cnt, median=False):
    net.eval()
    net.config.NUM_RULE = rule_cnt
    net.config.UPDATE = True

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    timer.tic()

    if median:
        loss_total = [[[] for _ in range(rule_cnt)] for __ in range(600)]
    else:
        loss_total = np.zeros([600, rule_cnt])

    for i, batch in enumerate(loader):
        # batch["rule"] (bz, 18, 10, k), batch["gt_range"] (bz, 18), rule (600, 10, k)
        rule = rule_set[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()
        if config.MODEL.get('DYNAMIC', False):
            batch["rule_cnt"] = torch.from_numpy(rule_adaptive[batch["gt_range"]]).long()  # (600),(bz,18)->(bz,18)
        for key in ["gt_label", "gt_pvp", "gt_obj", "rule", "gt_range", "shuffle_index", 'gt_part', "rule_cnt",
                    'FP0l', 'FP0r',
                    'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5',
                    'FO', 'FH', 'FR', 'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5',
                    'labels_r', 'rule_lens']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)

        # batch["gt_range"] (bz,18)
        gt_range = batch["gt_range"].detach().cpu().numpy()
        for j in range(rule_cnt):
            loss_cur = output["L_cls" + str(j)].detach().cpu().numpy()  # (bz,18)
            if median:
                for a in range(loss_cur.shape[0]):
                    for b in range(loss_cur.shape[1]):
                        loss_total[gt_range[a, b]][j].append(loss_cur[a, b])
            else:
                loss_total[gt_range, j] += loss_cur

        timer.toc()
        timer.tic()

        if i % 100 == 0:
            print(
                "%03d epoch eval rule, %05d iter, average time %.4f"
                % (epoch, i, timer.average_time)
            )
    timer.toc()

    if median:
        loss_median = np.zeros([600, rule_cnt])
        for k in range(600):
            for j in range(rule_cnt):
                cur = np.argsort(np.array(loss_total[k][j]))
                loss_median[k, j] = cur[int(len(cur) / 2)]
        return net, loss_median

    return net, loss_total


def eval_with_separate_rule_by_train_accuracy(net, loader, timer, epoch, rule_set, rule_cnt):
    net.eval()
    net.config.NUM_RULE = rule_cnt
    net.config.UPDATE = True

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    timer.tic()

    cnt_total = torch.from_numpy(np.zeros([600, rule_cnt]))
    cnt_correct = torch.from_numpy(np.zeros([600, rule_cnt]))

    for i, batch in enumerate(loader):
        # batch["rule"] (bz, 18, 10, k), batch["gt_range"] (bz, 18), rule (600, 10, k)
        rule = rule_set[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()

        for key in ["gt_label", "gt_pvp", "gt_obj", "rule", "gt_range", "shuffle_index", 'gt_part',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5',
                    'FO', 'FH', 'FR', 'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5',
                    'labels_r', 'rule_lens']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)

        gt_range = batch["gt_range"].detach().cpu()
        gt_label = batch["gt_label"].detach().cpu()
        for j in range(rule_cnt):
            prob = output["p" + str(j)].detach().cpu().numpy()
            correctness = (prob > 0.5) == gt_label
            cnt_correct[gt_range, j] += correctness
            cnt_total[gt_range, j] += 1

        timer.toc()
        timer.tic()

        if i % 100 == 0:
            print(
                "%03d epoch eval rule, %05d iter, average time %.4f"
                % (epoch, i, timer.average_time)
            )
    timer.toc()

    cnt_total = cnt_total.numpy()
    cnt_correct = cnt_correct.numpy()
    acc = cnt_correct / cnt_total

    return net, acc


def eval_with_separate_rule(net, loader, timer, epoch, rule_set, rule_cnt):
    net.config.NUM_RULE = rule_cnt
    net.config.UPDATE = True

    net.eval()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    bboxes, scores, keys = [], [], []
    for i in range(80):
        bboxes.append([])
        scores.append([])
        keys.append([])

    scores_sepa = [[[] for __ in range(80)] for _ in range(rule_cnt)]

    timer.tic()

    for i, batch in enumerate(loader):
        rule = rule_set[batch["gt_range"]]  # (600,88,k), (bz,18) -> (bz,18,88,k)
        batch["rule"] = torch.from_numpy(rule).long()

        for key in ["gt_label", "gt_pvp", "gt_obj", "rule", "gt_range", "shuffle_index", 'gt_part',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5',
                    'FO', 'FH', 'FR', 'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5',
                    'labels_r', 'rule_lens']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)

        bbox = batch["spatial"]

        obj_class = batch["obj_class"]
        hdet = batch["hdet"]
        odet = batch["odet"]

        for j in range(bbox.shape[0]):
            cls = obj_class[j]
            x, y = obj_range[cls][0] - 1, obj_range[cls][1]
            keys[cls].append(batch["key"][j])
            bboxes[cls].append(bbox[j])

        for k in range(rule_cnt):
            prob = output["p" + str(k)].detach().cpu().numpy()

            for j in range(bbox.shape[0]):
                cls = obj_class[j]
                x, y = obj_range[cls][0] - 1, obj_range[cls][1]

                if cls in obj_range_extra:
                    p = prob[j, -obj_range_cnt[cls]:]
                else:
                    p = prob[j, : obj_range_cnt[cls]]

                if not config.TEST.get('LIS') or config.TEST.LIS:
                    scores_sepa[k][cls].append(
                        p * getSigmoid(9, 1, 3, 0, hdet[j]) * getSigmoid(9, 1, 3, 0, odet[j])
                    )
                else:
                    scores_sepa[k][cls].append(p * hdet[j] * odet[j])

        timer.toc()
        timer.tic()

        if i % 100 == 0:
            print(
                "%03d epoch, %05d iteration, average time %.4f"
                % (epoch, i, timer.average_time)
            )
    timer.toc()

    for i in range(80):
        keys[i] = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])

    map_sepa = []

    for k in range(rule_cnt):
        scores_cur = scores_sepa[k]
        for i in range(80):
            scores_cur[i] = np.array(scores_cur[i])
        map_cur, _ = get_map(keys, scores_cur, bboxes)
        map_sepa.append(map_cur)

    return map_sepa


def eval_train_map(net, loader, timer, epoch, rule_set, rule_cnt):
    net.eval()
    net.config.NUM_RULE = rule_cnt
    net.config.UPDATE = False

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    labels, scores = [], []
    for i in range(80):
        labels.append([])
        scores.append([])

    timer.tic()

    for i, batch in enumerate(loader):

        rule = rule_select[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()
        if config.MODEL.get('DYNAMIC', False):
            batch["rule_cnt"] = torch.from_numpy(rule_adaptive[batch["gt_range"]]).long()  # (600),(bz,18)->(bz,18)
        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part', 'gt_label',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)
        gt_obj = batch["gt_obj"].detach().cpu().numpy()
        gt_label = batch["gt_label"].detach().cpu().numpy()
        prob = output["p"].detach().cpu().numpy()

        for j in range(gt_obj.shape[0]):
            cls = np.where(gt_obj[j])[0][0]
            x, y = obj_range[cls][0] - 1, obj_range[cls][1]
            labels[cls].append(np.where(gt_label[j])[0])

            if cls in obj_range_extra:
                p = prob[j, -obj_range_cnt[cls]:]
            else:
                p = prob[j, : obj_range_cnt[cls]]

            scores[cls].append(p)

        timer.toc()
        timer.tic()
        if i % 100 == 0:
            print(
                "%03d epoch, %05d iteration, average time %.4f"
                % (epoch, i, timer.average_time)
            )
    timer.toc()

    for i in range(80):
        scores[i] = np.array(scores[i])
        labels[i] = np.array(labels[i])

    with open(os.path.join(cur_path, 'result.pkl'), 'wb') as f:
        pickle.dump({'scores': scores, 'labels': labels}, f)

    return net


def eval_with_separate_rule_by_train_map(net, loader, timer, epoch, rule_set, rule_cnt):
    net.eval()
    net.config.NUM_RULE = rule_cnt
    net.config.UPDATE = True

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(True)

    labels, scores = [], []
    for i in range(80):
        labels.append([])
        scores.append([])

    scores_sepa = [[[] for __ in range(80)] for _ in range(rule_all_cnt)]

    timer.tic()

    for i, batch in enumerate(loader):

        rule = rule_set[batch["gt_range"]]  # (600,10,k), (bz,18) -> (bz,18,10,k)
        batch["rule"] = torch.from_numpy(rule).long()
        if config.MODEL.get('DYNAMIC', False):
            batch["rule_cnt"] = torch.from_numpy(rule_adaptive[batch["gt_range"]]).long()  # (600),(bz,18)->(bz,18)
        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part', 'gt_label', 'rule_cnt',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        output = net(batch)
        gt_obj = batch["gt_obj"].detach().cpu().numpy()
        gt_label = batch["gt_label"].detach().cpu().numpy()
        # prob = output["p"].detach().cpu().numpy()

        for j in range(gt_obj.shape[0]):
            cls = np.where(gt_obj[j])[0][0]
            labels[cls].append(np.where(gt_label[j])[0])

        for k in range(rule_cnt):
            prob = output["p" + str(k)].detach().cpu().numpy()

            for j in range(gt_obj.shape[0]):
                cls = np.where(gt_obj[j])[0][0]
                x, y = obj_range[cls][0] - 1, obj_range[cls][1]
                # labels[cls].append(np.where(gt_label[j])[0])

                if cls in obj_range_extra:
                    p = prob[j, -obj_range_cnt[cls]:]
                else:
                    p = prob[j, : obj_range_cnt[cls]]

                scores_sepa[k][cls].append(p)
                # scores[cls].append(p)

        timer.toc()
        timer.tic()
        if i % 100 == 0:
            print(
                "%03d epoch, %05d iteration, average time %.4f"
                % (epoch, i, timer.average_time)
            )
    timer.toc()

    for i in range(80):
        # scores[i] = np.array(scores[i])
        labels[i] = np.array(labels[i])

    map_sepa = []

    for k in range(rule_cnt):
        scores_cur = scores_sepa[k]
        for i in range(80):
            scores_cur[i] = np.array(scores_cur[i])
        map_cur, _ = get_train_map(scores_cur, labels)
        map_sepa.append(map_cur)

    # with open(os.path.join(cur_path, 'result.pkl'), 'wb') as f:
    #     pickle.dump({'scores': scores, 'labels': labels}, f)

    return net, map_sepa


def select_rule_by_train_loss(rule_all, rule_select, loss_total, loss_select, measure_map=False, rule_adaptive=None):
    if measure_map:
        loss_total = np.array(loss_total[1]).transpose()  # (88,600)->(600,88)
        loss_select = np.array(loss_select[1]).transpose()  # (10,600)->(600,10)
    else:
        loss_total = np.array(loss_total[1])  # negative, the bigger, the better
        loss_select = np.array(loss_select[1])

    if rule_adaptive is not None:
        for j in range(rule_all.shape[0]):
            for k in range(rule_all.shape[1]):
                if np.sum(rule_all[j, k]) == 0:
                    break
                if loss_total[j, k] > np.max(loss_select[j]):
                    if rule_adaptive[j] < config.MODEL.NUM_RULE:
                        rule_select[j, int(rule_adaptive[j])] = np.array(rule_all[j, k])
                        loss_select[j, int(rule_adaptive[j])] = loss_total[j, k]
                        rule_adaptive[j] += 1
                    else:
                        # min_index = np.where(np.sum(rule_select[j]) == 0)[0][0]
                        # if len(min_index) == 0:
                        #     min_index = np.argmin(loss_select[j])
                        min_index = np.argmin(loss_select[j])
                        rule_select[j, min_index] = np.array(rule_all[j, k])
                        loss_select[j, min_index] = loss_total[j, k]
    else:
        for j in range(rule_all.shape[0]):
            for k in range(rule_all.shape[1]):
                if np.sum(rule_all[j, k]) == 0:
                    break
                if loss_total[j, k] > np.max(loss_select[j]):
                    min_index = np.argmin(loss_select[j])
                    rule_select[j, min_index] = np.array(rule_all[j, k])
                    loss_select[j, min_index] = loss_total[j, k]

    return rule_select, rule_adaptive


# if args.eval:
#     map = eval(net, test_loader, test_timer, 0, eval=True)
#     print(np.mean(map))
#     quit()

bst = 0.0

if args.eval:
    map_sepa = eval_with_separate_rule(net, test_loader, test_timer, 0, rule_select, rule_select_cnt)
    with open(os.path.join(cur_path, "map_sepa.pkl"), 'wb') as f:
        pickle.dump(map_sepa, f)
    print([100 * np.mean(map_sepa[i]) for i in range(len(map_sepa))])
    quit()

if args.eval_train_map:
    net, map_sepa = eval_with_separate_rule_by_train_map(net, train_loader, train_timer, 0, rule_select,
                                                         rule_select_cnt)
    print([100 * np.mean(map_sepa[i]) for i in range(len(map_sepa))])
    quit()

loss_select = None

# loss_total = pickle.load(open(os.path.join(cur_path, 'loss_total.pkl'), 'rb'))
# loss_select = pickle.load(open(os.path.join(cur_path, 'loss_select.pkl'), 'rb'))
# rule_select, rule_adaptive = select_rule_by_train_loss(rule_all, rule_select, loss_total, loss_select,
#                                                                measure_map=(config.MODEL.MEASURE == 'train_map'),
#                                                                rule_adaptive=rule_adaptive)

for i in range(config.TRAIN.MAX_EPOCH):
    # if not args.debug:
    if i > 0:
        net, train_meters = train(net, train_loader, optimizer, train_timer, i, rule_select)
        if "LR_SCHEDULER" in config.TRAIN:
            scheduler.step()
        train_str = "%03d epoch training" % i

        for (key, value) in train_meters.items():
            train_str += ", %s=%.4f" % (key, value.avg)
        logger.info(train_str)

    map = eval(net, test_loader, test_timer, i, rule_select)
    ap = np.mean(map)
    test_str = "%03d epoch evaluation, mAP=%.2f" % (i, ap * 100)
    logger.info(test_str)

    if i % config.MODEL.EPOCH_RATIO == 0:
        if config.MODEL.MEASURE == 'train_loss_sum':
            loss_total = eval_with_separate_rule_by_train_loss(net, update_loader, eval_timer, i, rule_all,
                                                               rule_all_cnt)
        elif config.MODEL.MEASURE == 'train_loss_median':
            loss_total = eval_with_separate_rule_by_train_loss(net, update_loader, eval_timer, i, rule_all,
                                                               rule_all_cnt, median=True)
        elif config.MODEL.MEASURE == 'train_acc':
            loss_total = eval_with_separate_rule_by_train_accuracy(net, update_loader, eval_timer, i, rule_all,
                                                                   rule_all_cnt)
        elif config.MODEL.MEASURE == 'train_map':
            loss_total = eval_with_separate_rule_by_train_map(net, update_loader, eval_timer, i, rule_all,
                                                              rule_all_cnt)
        else:
            raise NotImplementedError
        with open(os.path.join(cur_path, 'loss_total.pkl'), 'wb') as f:
            pickle.dump(loss_total, f)

        if config.MODEL.MEASURE == 'train_loss_sum':
            loss_select = eval_with_separate_rule_by_train_loss(net, update_loader, eval_timer, i, rule_select,
                                                                rule_select_cnt)
        elif config.MODEL.MEASURE == 'train_loss_median':
            loss_select = eval_with_separate_rule_by_train_loss(net, update_loader, eval_timer, i, rule_select,
                                                                rule_select_cnt, median=True)
        elif config.MODEL.MEASURE == 'train_acc':
            loss_select = eval_with_separate_rule_by_train_accuracy(net, update_loader, eval_timer, i, rule_select,
                                                                    rule_select_cnt)
        elif config.MODEL.MEASURE == 'train_map':
            loss_select = eval_with_separate_rule_by_train_map(net, update_loader, eval_timer, i, rule_select,
                                                               rule_select_cnt)
        else:
            raise NotImplementedError
        with open(os.path.join(cur_path, 'loss_select.pkl'), 'wb') as f:
            pickle.dump(loss_select, f)

    # else:
    #     loss_total = pickle.load(open(os.path.join(cur_path, 'loss_total.pkl'), 'rb'))
    #     loss_select = pickle.load(open(os.path.join(cur_path, 'loss_select.pkl'), 'rb'))
    #     ap = 0.0

    try:
        state = {
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.config,
            "map": map,
            "step": step,
        }
    except:
        state = {
            "state": net,
            "optim_state": optimizer.state_dict(),
            "config": net.module.config,
            "map": map,
            "step": step,
        }
    if "LR_SCHEDULER" in config.TRAIN:
        state["lr_scheduler_state"] = scheduler.state_dict()

    if ap > bst:
        bst = ap
        torch.save(state, os.path.join(cur_path, "bst.pth"))
        with open(os.path.join(cur_path, "bst_select_rule.pkl"), 'wb') as f:
            pickle.dump(rule_select, f)

    # torch.save(state, os.path.join(cur_path, "latest.pth"))
    # with open(os.path.join(cur_path, "select_rule_%d.pkl" % i), 'wb') as f:
    #     pickle.dump(rule_select, f)

    if i % config.MODEL.EPOCH_RATIO == 0:
        rule_select, rule_adaptive = select_rule_by_train_loss(rule_all, rule_select, loss_total, loss_select,
                                                               measure_map=(config.MODEL.MEASURE == 'train_map'),
                                                               rule_adaptive=rule_adaptive)
