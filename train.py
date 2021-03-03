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
    NLR_10v_simplified_no_T, part_classifier, NLR_simplified_no_T
)
from utils import Timer, HO_weight, AverageMeter
from HICO_DET_utils import obj_range, get_map, getSigmoid
from dataset import (
    HICO_train_set,
    HICO_test_set,
    HICO_gt_test_set
)
from object_to_hoi import obj_range_padding, obj_range_cnt, obj_range_extra
from pasta_utils import pvp_weight


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


optims = {}
optims["RMSprop"] = optim.RMSprop
optims["SGD"] = optim.SGD
optims["AdamW"] = optim.AdamW

# export CUDA_VISIBLE_DEVICES=2; python train_nlr.py --config_path configs/default_nlr_pvp.yml
# watch -n 1 'grep mAP exp/judger_avg_pvp/log.txt | tail'

gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
device = torch.device("cuda:{}".format(gpus[0]))

models = {
    # "NLR": NLR,
    # "Linear": Linear,
    # "NLR_simplified": NLR_simplified,
    # "NLR_10v": NLR_10v,
    # "NLR_simplified_no_T": NLR_simplified_no_T,
    # "VL_align": VL_align,
    # "Linear_10v": Linear_10v,
    "NLR_10v_simplified_no_T": NLR_10v_simplified_no_T,
    "NLR_simplified_no_T": NLR_simplified_no_T
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
        "--eval_for_store",
        dest="eval_for_store",
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

config = get_config(args)
config, ori_ckpt = get_resume(config)
yaml.dump(dict(config), open(os.path.join(cur_path, "config.yml"), "w"))

if config.MODE != "Linear":
    set_seed(config.RANDOM_SEED)

logger, writer = get_logger(cur_path)
logger.info("Start print log")

if not args.eval:
    train_set = HICO_train_set(config, split='trainval', train_mode=True)

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

if not config.get("GT_TEST", False):
    test_set = HICO_test_set(config, split='test')
else:
    test_set = HICO_gt_test_set(config, split='test')

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

if config.MODEL.get('PART_CLS'):
    part_classifier = part_classifier(config.MODEL.PART_CLS, pvp_weight)
    if len(gpus) > 1:
        part_classifier = torch.nn.DataParallel(part_classifier.to(device), device_ids=gpus, output_device=gpus[0])
    else:
        part_classifier = part_classifier.to(device)
else:
    part_classifier = None

logger.info(net)
if ori_ckpt or args.eval:
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
if ori_ckpt and config.TRAIN.OPTIMIZER.RESUME:
    optimizer.load_state_dict(ori_ckpt["optim_state"])

train_timer, test_timer = Timer(), Timer()

cur_epoch = 0
bst = 0.0
step = 0


def train(net, loader, optimizer, timer, epoch, part_classifier=None):
    net.train()
    if part_classifier:
        if config.MODEL.PART_CLS.FROZEN:
            part_classifier.eval()
        else:
            part_classifier.train()

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
            'rule_lens'
        ]:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        if part_classifier:
            if config.MODE == 'NLR_10v_simplified_no_T':
                batch["gt_part"] = torch.sigmoid(part_classifier(batch)['s_10v'])
            elif config.MODE == 'NLR_simplified_no_T':
                batch["gt_pvp"] = torch.sigmoid(part_classifier(batch)['s'])
            else:
                raise NotImplementedError

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


def eval(net, loader, timer, epoch, part_classifier=None, eval=False):
    net.eval()
    part_classifier.eval()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(False)

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

    for i, batch in enumerate(loader):

        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        if part_classifier:
            if config.MODE == 'NLR_10v_simplified_no_T':
                batch["gt_part"] = torch.sigmoid(part_classifier(batch)['s_10v'])
            elif config.MODE == 'NLR_simplified_no_T':
                batch["gt_pvp"] = torch.sigmoid(part_classifier(batch)['s'])
            else:
                raise NotImplementedError

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

            # if (
            #         config.MODE in ["Linear", "Linear_10v"]
            #         or config.MODEL.NUM_CLASS_SELECT == 600
            #         or config.MODEL.NUM_CLASS == 117
            # ):
            #     p = prob[j, x:y]
            # else:

            if cls in obj_range_extra:
                p = prob[j, -obj_range_cnt[cls]:]
                s = sc[j, -obj_range_cnt[cls]:]
            else:
                p = prob[j, : obj_range_cnt[cls]]
                s = sc[j, : obj_range_cnt[cls]]

            if config.TEST.LIS:
                scores[cls].append(
                    p * getSigmoid(9, 1, 3, 0, hdet[j]) * getSigmoid(9, 1, 3, 0, odet[j])
                )
            else:
                scores[cls].append(p * hdet[j] * odet[j])

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
    timer.toc()

    for i in range(80):
        keys[i] = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])

        if eval:
            hdets[i] = np.array(hdets[i])
            odets[i] = np.array(odets[i])
            scores_wo_lis[i] = np.array(scores_wo_lis[i])

    map, mrec = get_map(keys, scores, bboxes)

    if eval:
        with open(os.path.join(cur_path, 'result.pkl'), 'wb') as f:
            pickle.dump({'keys': keys, 'scores': scores_wo_lis, 'bboxes': bboxes, 'hdet': hdets, 'odet': odets}, f)

    return map

def eval_for_store(net, loader, timer, epoch):
    net.eval()
    part_classifier.eval()

    if config.MODE not in ["Linear", "Linear_10v"]:
        net.set_status(False)

    bboxes, scores, keys, labels_r = [], [], [], []
    for i in range(80):
        bboxes.append([])
        scores.append([])
        keys.append([])
        labels_r.append([])

    timer.tic()

    for i, batch in enumerate(loader):

        for key in ["gt_pvp", "gt_obj", "rule", "gt_range", 'gt_part', 'rule_cnt',
                    'FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5', 'FO', 'FH', 'FR',
                    'FS',
                    'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5',
                    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'labels_r']:
            if key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)

        if part_classifier:
            if config.MODE == 'NLR_10v_simplified_no_T':
                batch["gt_part"] = torch.sigmoid(part_classifier(batch)['s_10v'])
            elif config.MODE == 'NLR_simplified_no_T':
                batch["gt_pvp"] = torch.sigmoid(part_classifier(batch)['s'])
            else:
                raise NotImplementedError

        output = net(batch)

        bbox = batch["spatial"]
        obj_class = batch["obj_class"]
        # label_r = batch["labels_r_"]

        prob = output["p"].detach().cpu().numpy()

        for j in range(bbox.shape[0]):
            cls = obj_class[j]
            x, y = obj_range[cls][0] - 1, obj_range[cls][1]
            keys[cls].append(batch["key"][j])
            bboxes[cls].append(bbox[j])
            # labels_r[cls].append(label_r[j])

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
        keys[i] = np.array(keys[i])
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])
        labels_r[i] = np.array(labels_r[i])
        
    with open(os.path.join(cur_path, 'result.pkl'), 'wb') as f:
        pickle.dump({'keys': keys, 'bboxes': bboxes, 'scores':scores, 'labels_r':labels_r}, f)

    map, mrec = get_map(keys, scores, bboxes)

    return map

if args.eval_for_store:
    map = eval_for_store(net, test_loader, test_timer, 0)
    print(np.mean(map))
    quit()

if args.eval:
    map = eval(net, test_loader, test_timer, 0, eval=True, part_classifier=part_classifier)
    print(np.mean(map))
    quit()

bst = 0.0

# map = eval(net, test_loader, test_timer, -1, part_classifier=part_classifier)

# if config.MODEL.CHECKPOINT:
#     map = eval(net, test_loader, test_timer, -1, part_classifier=part_classifier)
#     ap = np.mean(map)
#     test_str = "-1 epoch evaluation, mAP=%.2f" % (ap * 100)
#     logger.info(test_str)
#
#     try:
#         state = {
#             # 'state': net.state_dict(),
#             "state": net,
#             "optim_state": optimizer.state_dict(),
#             "config": net.config,
#             "map": map,
#             "step": step,
#         }
#     except:
#         state = {
#             # 'state': net.state_dict(),
#             "state": net,
#             "optim_state": optimizer.state_dict(),
#             "config": net.module.config,
#             "map": map,
#             "step": step,
#         }
#     if ap > bst:
#         bst = ap
#         torch.save(state, os.path.join(cur_path, "bst.pth"))
#     torch.save(state, os.path.join(cur_path, "latest.pth"))

if config.MODE == 'NLR_10v' and config.MODEL.NLR.CHECKPOINT:
    map = eval(net, test_loader, test_timer, -1, part_classifier=part_classifier)
    ap = np.mean(map)
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
    net, train_meters = train(net, train_loader, optimizer, train_timer, i, part_classifier=part_classifier)
    if "LR_SCHEDULER" in config.TRAIN:
        scheduler.step()
    train_str = "%03d epoch training" % i

    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    logger.info(train_str)

    map = eval(net, test_loader, test_timer, i, part_classifier=part_classifier)
    ap = np.mean(map)
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
