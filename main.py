# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import pickle
import random

from pathlib import Path

import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


import util.misc as utils
from datasets import build_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--data_root', default='data/hake', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--arch', default='ViT-B/16', type=str)
    parser.add_argument('--backbone_pretrained', default="pretrained/clip/ViT-B-16.new.pt", type=str)
    parser.add_argument('--backbone_freeze_layer', default=12, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--vis_test', default=False, action='store_true')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./runs/',
                        help='path where to save, empty for no saving')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def train_one_epoch(model, data_loader_train, lr_scheduler, optimizer, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    for img, gt_labels, img_metas in metric_logger.log_every(data_loader_train, print_freq, header):
        img = img.to(device)
        loss_dict = model(img, gt_labels, img_metas)
        losses = sum(loss_dict[k] for k in loss_dict.keys() if 'loss' in k)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {k: v.item() for k, v in loss_dict_reduced.items()}

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(losses=losses.detach().cpu().item())
        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def eval_one_epoch(model, data_loader_train, device, epoch, output_dir, vis):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    result = []
    for img, gt_labels, img_metas in metric_logger.log_every(data_loader_train, 500, header):
        img = img.to(device)

        out = model(img, gt_labels, img_metas)
        if vis:
            data_loader_train.dataset.vis_test(img_metas[0]['img_path'], out[0], output_dir)

        result += out

    rank = utils.get_rank()
    with open(f'{output_dir}/result_rank{rank}.pkl', 'wb') as f:
        pickle.dump(result, f)

    dist.barrier()
    if rank == 0:
        all_result = []
        for i in range(utils.get_world_size()):
            result = pickle.load(open(f'{output_dir}/result_rank{rank}.pkl', 'rb'))
            all_result += result
        data_loader_train.dataset.evaluate(all_result)
        print(f"finished evaluate total {len(all_result)} record")
    dist.barrier()


def main(args):
    utils.init_distributed_mode(args)

    device = args.device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    # print(f"Model: {model}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    print(f'train size {len(dataset_train)}')
    print(f'val size {len(dataset_val)}')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader_train)*args.epochs, eta_min=1e-6)

    output_dir = args.output_dir

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print(f'resume model from {args.resume}')
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        eval_one_epoch(model, data_loader_val, device, args.start_epoch, args.output_dir, args.vis_test)
        return

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(
            model, data_loader_train, lr_scheduler, optimizer, device, epoch)
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, osp.join(output_dir, f'ckpt_{epoch}.pth'))
        eval_one_epoch(
            model, data_loader_val, device, epoch, args.output_dir, args.vis_test
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
