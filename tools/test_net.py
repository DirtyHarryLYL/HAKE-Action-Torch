#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
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

from activity2vec.networks.pasta_net import pasta_res50
from activity2vec.dataset.hake_dataset import hake_test
from activity2vec.ult.config import get_cfg
from activity2vec.ult.logging import setup_logging
from activity2vec.ult.checkpoint import load_model
from activity2vec.ult.parser import test_parse_args
from benchmark import benchmark

def custom_collate(batch):
    image, annos, image_id = zip(*batch)
    image, annos = default_collate(image), default_collate(annos)
    return image, annos, image_id

@torch.no_grad()
def test(cfg, net, test_loader, output_dir, logger):
    logger.info("==> Testing Activity2Vec, output_dir: " + output_dir + ' ...')
    for image, annos, image_id in tqdm(test_loader, ncols=40):
        human_bboxes = annos['human_bboxes']
        if human_bboxes.shape[1] == 0:
            continue

        image = image.cuda(non_blocking=True)
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

        _, p_pasta, p_verb = net(image, annos)
        if cfg.TEST.HUMAN_SCORE_ENHANCE:
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

def setup():
    cfg = get_cfg()
    args = test_parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.TEST.WEIGHT_PATH  = os.path.abspath(cfg.TEST.WEIGHT_PATH)
    model_dir, model_filename = os.path.split(cfg.TEST.WEIGHT_PATH)
    model_name = os.path.split(model_dir)[-1]
    cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'results', model_name)
    cfg.TEST.OUTPUT_DIR = os.path.join(cfg.ROOT_DIR, 'results', model_name, model_filename+'_results')
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    cfg.freeze()
    return cfg, args

if __name__ == "__main__":
    cfg, args = setup()

    logger = setup_logging(cfg.LOG_DIR).test
    logger.info('==> cfg:')
    logger.info(cfg)

    if args.eval:
        os.makedirs(cfg.TEST.OUTPUT_DIR, exist_ok=True)

        # model preparing
        net = pasta_res50(cfg)
        net, _, _, _ = load_model(cfg, net, None, None, cfg.TEST.WEIGHT_PATH, mode='test')
        net.eval()
        net = net.cuda()
        test_loader = torch.utils.data.DataLoader(dataset=hake_test(cfg), 
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  collate_fn=custom_collate, 
                                                  num_workers=1)

        test(cfg, net, test_loader, cfg.TEST.OUTPUT_DIR, logger)
    
    if args.benchmark:
        benchmark(cfg.TEST.OUTPUT_DIR, cfg, logger)