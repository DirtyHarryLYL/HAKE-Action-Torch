#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Dec 3rd, 2020             #
#############################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data._utils.collate import default_collate

import argparse
from tqdm import tqdm
import os
import h5py
import pprint

from activity2vec.networks.pasta_net import pasta_res50
from activity2vec.dataset.hake_dataset import hake_test
from activity2vec.ult.config import get_cfg
from benchmark import benchmark

def print_and_log(message, cfg):
    print(message)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    with open(os.path.join(cfg.LOG_DIR, "test.log"), 'a') as f:
        f.write(message + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Testing Activity2Vec')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--weight', 
            help='the path of weight to load from',
            default="",
            type=str)
    parser.add_argument('--eval', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    parser.add_argument('--benchmark', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    parser.add_argument('--no-human-score', action='store_true')
    parser.add_argument('--no-part-box', action='store_true')
    parser.add_argument('--show-action-res', action='store_true')
    args = parser.parse_args()
    return args

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

def custom_collate(batch):
    image, annos, image_id = zip(*batch)
    image, annos = default_collate(image), default_collate(annos)
    return image, annos, image_id

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model

@torch.no_grad()
def test(cfg, net, test_loader, output_dir):
    print_and_log(("==> Testing Activity2Vec, output_dir: " + output_dir + ' ...'), cfg)
    for image, annos, image_id in tqdm(test_loader, ncols=40):
        human_bboxes = annos['human_bboxes']
        if human_bboxes.shape[1] == 0:
            continue
        image = image.cuda(non_blocking=True)
        annos = annos_to_cuda(cfg, annos)
        _, p_pasta, p_verb = net(image, annos)
        if cfg.TRAIN.HUMAN_SCORE_ENHANCE:
            human_scores = annos['human_scores'].unsqueeze(-1)
            p_pasta *= human_scores
            p_verb *= human_scores
        p_pasta = p_pasta.detach().cpu().numpy()
        p_verb  = p_verb.detach().cpu().numpy()
        
        out_path = os.path.join(output_dir, image_id[0]+'.hdf5')
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        h = h5py.File(out_path, 'w')
        h.create_dataset('human_bboxes', data=human_bboxes)
        h.create_dataset('pasta_score', data=p_pasta)
        h.create_dataset('verb_score', data=p_verb)
        h.close()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg = get_cfg()
    cfg.WEIGHT_PATH  = os.path.abspath(args.weight)
    model_dir, model_filename = os.path.split(args.weight)
    model_name = os.path.split(model_dir)[-1]
    cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'results', model_name)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    output_dir = os.path.join(cfg.ROOT_DIR, 'results', model_name, model_filename+'_results')
    cfg.TRAIN.HUMAN_SCORE_ENHANCE = not args.no_human_score
    cfg.BENCHMARK.SHOW_ACTION_RES = args.show_action_res
    if args.no_part_box:
        cfg.MODEL.PART_ROI_ENABLE = False
    print_and_log('==> cfg:\n'+str(pprint.pformat(cfg, indent=2)), cfg)
    if args.eval:
        os.makedirs(output_dir, exist_ok=True)

        # model preparing
        net = pasta_res50(cfg)
        net = load_model(net, cfg.WEIGHT_PATH)
        net.eval()
        net = net.cuda()
        test_set    = hake_test(cfg)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=1)

        test(cfg, net, test_loader, output_dir)
    
    if args.benchmark:
        benchmark(output_dir, cfg)