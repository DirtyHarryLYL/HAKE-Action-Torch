#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Dec 8th, 2020             #
#############################################
import argparse
import os
import os.path as osp
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__),'..','utils')))
from custom_multiprocessing import process_pool
from part_box_generation import output_part_box, map_17_to_16
from pasta_inference import pasta_model
generator_dict = {
    'fast_res50_256x192':['configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                          'pretrained_models/fast_res50_256x192.pth',
                          'yolo']
    }

def args_parser():

    parser = argparse.ArgumentParser(description='Pose Data Generator')

    parser.add_argument('--alphapose', type=int, dest='alpha', default=1,
                        help='choose whether to use alphapose to generate pose coordinates')
    parser.add_argument('--partbox', type=int, dest='part', default=1,
                        help='choose whether to generate partboxes')
    parser.add_argument('--pasta', type=int, dest='pasta', default=1,
                        help='choose whether to output part states')

    parser.add_argument('--gpu', type=str, dest='gpu', default="0",
                        help='choose which cuda device to use')
    parser.add_argument('--generator', type=str, dest='generator', default='fast_res50_256x192',
                        help='choose which pose generator to use')
    parser.add_argument('--pasta-model', type=str, dest='pasta_weights', default='',
                        help='the path of pretrained pasta model')

    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")

    return parser.parse_args()

if __name__ == "__main__":

    args = args_parser()

    indir = osp.abspath(args.inputpath)
    outdir = osp.abspath(osp.join(indir, '..', osp.split(indir)[-1]+'-pose-pkl'))
    out_pkl_path = osp.abspath(osp.join(outdir, 'alphapose-results.pkl'))
    
    # input image --> pose keypoints
    if args.alpha:
        print('********AlphaPose Pose Detector********')
        alpha_generator = args.generator
        assert alpha_generator in generator_dict, 'generator {:s} not supported!'.format(alpha_generator)
        
        alpha_cfg, alpha_ckpt, alpha_detector = generator_dict[alpha_generator]
        cmd = ('python -u {binary} --gpus {gpus} '
            '--cfg {alphacfg} --checkpoint {alphackpt} '
            '--detector {alphadetector} --sp '
            '--indir {indir} --outdir {outdir} ')

        cmd = cmd.format(binary='scripts/demo_inference.py',
                        gpus=args.gpu,
                        alphacfg=alpha_cfg,
                        alphackpt=alpha_ckpt,
                        alphadetector=alpha_detector,
                        indir=indir,
                        outdir=outdir)
        cmd_cwd_list = [
                            (cmd,'./AlphaPose')
                       ]

        print('AlphaPose processing, processes num: {:d}...'.format(len(cmd_cwd_list)))

        alpha_process_pool = process_pool()
        alpha_process_pool.apply(cmd_cwd_list)
        alpha_process_pool.wait()

        cmd_cwd_list = []

        print('AlphaPose processed, pkl path: {:s}'.format(out_pkl_path))

    # pose keypoints --> part boxes
    if args.part:
        print('********Part Boxes Generation********')
        print('Part boxes generating...')
        out_pkl_with_part_boxes_path = osp.join(outdir, 'alphapose-results-with-part-boxes.pkl')
        pose_results = pickle.load(open(out_pkl_path,'rb'))
        for idx, each_human in enumerate(pose_results):
            keypoints = np.array(each_human['keypoints']).reshape(17,3)
            box = np.array(each_human['box'])
            box[2:] += box[:2]
            part_boxes = output_part_box(map_17_to_16(keypoints), box)
            pose_results[idx]['image_id'] = each_human['image_id']
            pose_results[idx]['keypoints'] = keypoints
            pose_results[idx]['box'] = box
            pose_results[idx]['part_boxes'] = part_boxes
        pickle.dump(pose_results, open(out_pkl_with_part_boxes_path,'wb'))
        print('Part boxes generated, pkl path: {:s}'.format(out_pkl_with_part_boxes_path))

    # part boxes --> pasta feature
    if args.pasta:
        print('********PaSta Inference********')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if not args.part:
            print('Loading generated part boxes...')
            out_pkl_with_part_boxes_path = osp.join(outdir, 'alphapose-results-with-part-boxes.pkl')
            pose_results = pickle.load(open(out_pkl_with_part_boxes_path,'rb'))
            print('Loaded.')
        out_pkl_pasta_path = osp.join(outdir, 'part-states-results.pkl')
        print('Constructing pasta inference model...')
        pasta_instance = pasta_model(args.pasta_weights)
        print('Finished.')
        print('Generating part states...')
        all_pasta_results = []
        for each_human in tqdm(pose_results):
            pasta_results = pasta_instance.inference(each_human)
            torch.cuda.empty_cache()
            all_pasta_results.append(pasta_results)
        pickle.dump(all_pasta_results, open(out_pkl_pasta_path,'wb'))
        print('Part states generated, pkl path:', out_pkl_pasta_path)