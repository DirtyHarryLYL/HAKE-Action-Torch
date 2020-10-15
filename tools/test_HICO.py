from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import logging
import time
import glob
from tqdm import tqdm
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from hakeaction.models.DJRN import DJRN_ResNet50
from hakeaction.datasets.HICO_dataset import HICO_Testset
from hakeaction.configs.config_DJRN import cfg
from datasets.utils.HICO_DET_utils import obj_range, get_map, get_map_with_NIS, getSigmoid, Get_next_sp_with_pose

def parse_args():
    parser = argparse.ArgumentParser(description='Train PVP on HICO')
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float) 

    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)

    parser.add_argument('--weight', 
            help='the path of weight to load from',
            default="",
            type=str)

    parser.add_argument('--stage', 
            help='specific stage for evaluation',
            default=1,
            type=int)

    args = parser.parse_args()
    return args

def load_model(model, ckp_path):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def test_HICO(net, test_loader, Test_RCNN, human_thres, object_thres, keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg, scores_HO):

    for test_data in tqdm(test_loader):

        image, image_id     = test_data['image'],  int(test_data['image_id'])
        Human_outs, Objects = Test_RCNN[image_id], Test_RCNN[image_id]
        This_image = []
        blobs = {}
        blobs['image'] = image
        blobs['att_2D_map'] = torch.randn(1, 17, 64, 64).cuda()

        for Human_out in Human_outs:
            if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
                
                blobs['H_num']   = 0
                blobs['H_boxes'] = [np.empty((0, 5), np.float64)]
                blobs['O_boxes'] = [np.empty((0, 5), np.float64)]
                blobs['sp']      = [np.empty((0, 64, 64, 3), np.float64)]
                blobs['gt_class_O'] = [np.empty((0, 80), np.float64)]

                H_box = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
                index = []

                for i,Object in enumerate(Objects):

                    if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object

                        # 1.the object detection result should > thres  
                        # 2.the bbox detected is not an object
                        if not (np.all(Object[2] == Human_out[2])): # This is a valid object
                            blobs['H_boxes'].append(H_box)
                            blobs['O_boxes'].append(np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5))
                            blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3))
                            classid = Object[4] - 1
                            tmp = np.zeros((1, 80), np.float64)
                            tmp[0, classid] = 1
                            blobs['gt_class_O'].append(tmp)
                            blobs['H_num'] += 1
                            index.append(i)

                blobs['H_boxes']    = np.concatenate(blobs['H_boxes'], axis=0)
                blobs['O_boxes']    = np.concatenate(blobs['O_boxes'], axis=0)
                blobs['sp']         = np.concatenate(blobs['sp'], axis=0)
                blobs['gt_class_O'] = np.concatenate(blobs['gt_class_O'], axis=0)

                if blobs['H_num'] == 0:
                    continue

                # move data to device(GPU)
                for k, v in blobs.items():
                    if k in ['image_id', 'H_num', 'image', 'att_2D_map']:
                        continue

                    blobs[k] = torch.from_numpy(v).float().cuda()

                # inference, test forward 
                cls_prob_HO, cls_prob_H, cls_prob_O, cls_prob_sp = net(blobs)
                cls_prob_HO  = cls_prob_HO.cpu().detach().numpy()
                cls_prob_H   = cls_prob_H.cpu().detach().numpy()
                cls_prob_O   = cls_prob_O.cpu().detach().numpy()
                cls_prob_sp  = cls_prob_sp.cpu().detach().numpy()
                # cls_prob_binary = cls_prob_binary.cpu().detach().numpy()

                for i in range(blobs['H_num']):
                    Object = Test_RCNN[image_id][index[i]]
                    classid = Object[4] - 1

                    keys[classid].append(image_id)
                    scores_H[classid].append(
                            cls_prob_H[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_O[classid].append(
                            cls_prob_O[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_sp[classid].append(
                            cls_prob_sp[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    hdet[classid].append(np.max(Human_out[5]))
                    odet[classid].append(np.max(Object[5]))
                    # pos[classid].append(cls_prob_binary[i][0])
                    # neg[classid].append(cls_prob_binary[i][1])
                    hbox = np.array(Human_out[2]).reshape(1, -1)
                    obox = np.array(Object[2]).reshape(1, -1)
                    bboxes[classid].append(np.concatenate([hbox, obox], axis=1))     

                    scores_HO[classid].append(
                        # with lis
                        # cls_prob_HO[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                        # getSigmoid(9, 1, 3, 0, Human_out[5]) * \
                        # getSigmoid(9, 1, 3, 0, Object[5])
                        
                        # without lis
                        cls_prob_HO[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                        Human_out[5] * \
                        Object[5]
                        )

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    cfg.WEIGHT_PATH  = args.weight
    args.model = args.weight.split('/')[-1]
    np.random.seed(cfg.RNG_SEED)

    output_dir  = cfg.ROOT_DIR + '/results/' + args.weight.split('/')[-1].strip('.tar') + '_Htre-{}_Otre-{}'.format(args.human_thres, args.object_thres)
    print("==> [test HICO] output_dir: ", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.stage == 1:
        # data preparing
        Test_RCNN   = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb" ), encoding="bytes")
        testImage   = glob.glob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg')
        test_set    = HICO_Testset(testImage, Test_RCNN, cfg)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

        # output holder preparing
        keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg, scores_HO= [], [], [], [], [], [], [], [], [], []
        for i in range(80):
            keys.append([])
            scores_H.append([])
            scores_O.append([])
            scores_sp.append([])
            scores_HO.append([])
            bboxes.append([])
            hdet.append([])
            odet.append([])
            pos.append([])
            neg.append([])

        # model preparing
        net = DJRN_ResNet50()
        net = load_model(net, cfg.WEIGHT_PATH)
        net.testMode = True
        net.eval()
        net = net.cuda()

        # proceed testing
        test_HICO(net, test_loader, Test_RCNN, args.human_thres, args.object_thres, keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg, scores_HO)

        # save test results
        for i in range(80):
            scores_H[i]  = np.concatenate(scores_H[i], axis=0)
            scores_O[i]  = np.concatenate(scores_O[i], axis=0)
            scores_sp[i] = np.concatenate(scores_sp[i], axis=0)
            scores_HO[i] = np.concatenate(scores_HO[i], axis=0) 
            bboxes[i]    = np.concatenate(bboxes[i], axis=0)
            keys[i]      = np.array(keys[i])
            hdet[i]      = np.array(hdet[i])
            odet[i]      = np.array(odet[i])
            pos[i]       = np.array(pos[i])
            neg[i]       = np.array(neg[i])
        
        # dump to pkl files
        pickle.dump(scores_H,  open(os.path.join(output_dir, 'score_H.pkl'), 'wb'))
        pickle.dump(scores_O,  open(os.path.join(output_dir, 'score_O.pkl'), 'wb'))
        pickle.dump(scores_sp, open(os.path.join(output_dir, 'score_sp.pkl'), 'wb'))
        pickle.dump(scores_HO, open(os.path.join(output_dir, 'scores_HO.pkl'), 'wb'))
        pickle.dump(bboxes,    open(os.path.join(output_dir, 'bboxes.pkl'), 'wb'))
        pickle.dump(keys,      open(os.path.join(output_dir, 'keys.pkl'), 'wb'))
        pickle.dump(hdet,      open(os.path.join(output_dir, 'hdet.pkl'), 'wb'))
        pickle.dump(odet,      open(os.path.join(output_dir, 'odet.pkl'), 'wb'))
        pickle.dump(pos,       open(os.path.join(output_dir, 'pos.pkl'), 'wb'))
        pickle.dump(neg,       open(os.path.join(output_dir, 'neg.pkl'), 'wb'))

    # stage 2: load pkl files to calculate mAP and mRecall
    if args.stage == 2:

        scores_H     = pickle.load(open(os.path.join(output_dir, 'score_H.pkl'), 'rb'))
        scores_O     = pickle.load(open(os.path.join(output_dir, 'score_O.pkl'), 'rb'))
        scores_sp    = pickle.load(open(os.path.join(output_dir, 'score_sp.pkl'), 'rb'))
        scores_HO    = pickle.load(open(os.path.join(output_dir, 'scores_HO.pkl'), 'rb')) 
        bboxes       = pickle.load(open(os.path.join(output_dir, 'bboxes.pkl'), 'rb'))
        keys         = pickle.load(open(os.path.join(output_dir, 'keys.pkl'), 'rb'))
        hdet         = pickle.load(open(os.path.join(output_dir, 'hdet.pkl'), 'rb'))
        odet         = pickle.load(open(os.path.join(output_dir, 'odet.pkl'), 'rb'))
        pos          = pickle.load(open(os.path.join(output_dir, 'pos.pkl'), 'rb'))
        neg          = pickle.load(open(os.path.join(output_dir, 'neg.pkl'), 'rb'))
    
    mAP, mrec = get_map(keys, scores_HO, bboxes, cfg)
    mAP, mrec = np.mean(mAP), np.mean(mrec)
    print("\n==> Evaluation Result without NIS: mAP={}, mrec={}".format(mAP, mrec))

    # NISThreshold = 0.9
    # mAP, mrec = get_map_with_NIS(keys, scores_HO, bboxes, pos, neg, NISThreshold)
    # mAP, mrec = np.mean(mAP), np.mean(mrec)
    # print("==> Evaluation  Result  with  NIS: mAP={}, mrec={}".format(mAP, mrec))
