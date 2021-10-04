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
import cv2
from tqdm import tqdm
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from lib.networks.TIN_HICO import TIN_ResNet50
from lib.dataset.HICO_dataset import HICO_Testset
from lib.ult.config_TIN import cfg
from lib.ult.ult import Get_next_sp_with_pose
from ult.HICO_DET_utils import obj_range, get_map, get_map_with_NIS, getSigmoid
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cnt = np.array([
    71,113,38,204,125,52,260,278,9,106,30,1380,124,150,26,17,116,89,1447,1183,1502,182,6,168,12,93,113,12,15,131,28,47,189,28,81,11,103,35,1382,339,182,997,594,34,8,106,159,200,541,55,13,20,27,195,137,12,532,30,29,43,789,431,10,5,107,31,6,393,67,79,5,28,35,484,149,306,9,8,197,101,13,95,65,10,6,40,43,117,92,1005,6,97,19,253,658,39,16,73,90,10,10,54,34,52,12,67,108,10,789,993,79,119,7,17,12,377,22,211,21,28,188,25,90,23,47,122,25,8,113,25,38,750,36,181,19,5,13,96,141,1253,440,1001,124,125,28,33,923,65,141,8,99,17,408,1131,1419,1313,278,18,8,138,110,23,552,314,81,10,6,58,11,249,14,14,11,61,25,54,53,171,39,7,30,6,119,34,6,92,77,137,8,7,292,163,10,143,24,9,132,88,10,26,61,239,41,30,35,7,7,22,914,57,23,18,983,89,14,139,10,129,321,117,14,24,7,104,53,90,810,6,407,7,716,7,194,194,6,23,33,51,8,7,345,156,37,19,107,1463,107,118,882,666,769,64,81,33,10,7,40,6,42,63,6,6,6,17,345,119,235,80,166,32,15,87,48,73,12,29,46,188,28,6,6,8,25,215,586,198,6,278,151,6,58,51,11,16,148,157,145,353,21,29,105,34,54,6,67,12,58,138,262,26,86,6,40,46,125,8,21,7,18,91,261,135,41,287,11,6,29,17,11,122,157,122,36,10,7,21,112,348,503,24,267,41,176,17,28,7,266,42,63,59,7,6,80,144,10,23,139,79,6,109,217,39,29,57,8,59,243,56,53,220,642,515,109,138,472,39,268,651,44,6,657,9,28,257,44,582,19,270,58,6,12,12,20,51,57,3,25,6,7,7,58,7,8,8,11,7,13,9,16,80,10,24,35,56,38,25,6,113,5,84,37,61,110,180,48,55,13,6,69,7,45,9,27,30,216,77,8,23,39,6,8,132,214,18,51,158,252,104,27,8,15,14,63,137,593,337,394,1185,58,1842,25,1480,50,11,52,83,25,141,34,11,546,581,716,13,10,42,74,180,344,604,602,657,10,290,115,6,48,17,64,220,109,139,251,381,105,321,27,83,6,6,168,14,28,14,4,32,227,130,228,6,14,13,34,31,6,97,231,11,304,21,9,29,6,240,181,18,6,61,47,240,68,8,35,186,241,6,112,14,71,6,108,26,42,42,659,67,8,7,6,6,11,6,8,19,68,8,10,108,252,351,6,17,15,19,16,15,121,25,183,47,79,25,206,154,17,74,889,1222,10,19,14,15,943,113,23,25,6,61,26,395,62,49,6,6,191,11,8,8,41,8
])

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

def test_HICO(net, cand, db, human_thres, object_thres, keys, scores, bboxes, hdet, odet):
    
    category_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, \
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
    
    for im_id in tqdm(cand.keys()):
    # for im_id in tqdm(list(cand.keys())[:100]):
        
        info     = db[im_id]
        filename = info['filename']
        key      = int(filename[-10:-4])
        im       = cv2.imread(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/' + filename)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)
        pairs    = info['pair_ids'][cand[im_id], :]
        cnt      = 0
        blobs    = {}
        image    = torch.from_numpy(im_orig).float().cuda().unsqueeze(0)
        blobs['image'] = image
        
        while cnt < len(pairs):
            if cnt + 75 <= len(pairs):
                r = cnt + 75
            else:
                r = len(pairs)
            blobs['H_num'] = r - cnt
            h_ind = pairs[cnt:r, 0]
            o_ind = pairs[cnt:r, 1]
            
            holder           = np.zeros((blobs['H_num'], 1))
            blobs['H_boxes'] = np.concatenate([holder, info['boxes'][h_ind, :]], axis=1)
            blobs['O_boxes'] = np.concatenate([holder, info['boxes'][o_ind, :]], axis=1)
            blobs['sp']      = []
            for i in range(blobs['H_num']):
                tmp_sp = Get_next_sp_with_pose(blobs['H_boxes'][i, 1:], blobs['O_boxes'][i, 1:], None)
                blobs['sp'].append(tmp_sp)
                # exit()
                # blobs['sp'].append(np.zeros((1, 64, 64, 3)))
                # blobs['sp'].append(np.zeros((64, 64, 3)))
            blobs['sp'] = np.array(blobs['sp'])
            
            # move data to device(GPU)
            for k, v in blobs.items():
                if k in ['image_id', 'H_num', 'image']:
                    continue
                blobs[k] = torch.from_numpy(v).float().cuda()
                    
            cls_prob_HO, cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary = net(blobs)
            cls_prob_HO     = cls_prob_HO.cpu().detach().numpy()
            cls_prob_H      = cls_prob_H.cpu().detach().numpy()
            cls_prob_O      = cls_prob_O.cpu().detach().numpy()
            cls_prob_sp     = cls_prob_sp.cpu().detach().numpy()
            cls_prob_binary = cls_prob_binary.cpu().detach().numpy()
            
            tmp_boxes   = np.concatenate([info['boxes'][h_ind, :], info['boxes'][o_ind, :]], axis=1)
            
            for i in range(blobs['H_num']):
                if info['obj_scores'][h_ind[i]] < human_thres:
                    continue
                classid = info['obj_classes'][o_ind[i]] - 1
                keys[classid].append(key)
                scores[classid].append(
                    cls_prob_HO[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                    info['obj_scores'][h_ind[i]] * \
                    info['obj_scores'][o_ind[i]])
                hdet[classid].append(info['obj_scores'][h_ind[i]])
                odet[classid].append(info['obj_scores'][o_ind[i]])
                # if info['obj_scores'][o_ind[i]]<0.1:
                #     print("O_score:{}".format(info['obj_scores'][o_ind[i]]))
                bboxes[classid].append(tmp_boxes[i:i+1, :])
                
            cnt = r

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    cfg.WEIGHT_PATH  = args.weight
    args.model = args.weight.split('/')[-1]
    np.random.seed(cfg.RNG_SEED)

    output_dir  = cfg.ROOT_DIR + '/-Results/PeyreTestData_' + args.weight.split('/')[-1].strip('.tar') + '_Htre-{}_Otre-{}'.format(args.human_thres, args.object_thres)
    print("==> [test HICO] output_dir: ", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.stage == 1:
        # data preparing
        # Using Peyre test data
        candidates = pickle.load( open( '/Disk5/yiming/TIN.torch.master/candidates_test.pkl', "rb" ), encoding="bytes")
        db         = pickle.load( open( '/home/zuoyu/TIN.torch/db_test.pkl', "rb" ), encoding="bytes")

        cand = {}
        for pair in candidates:
            [key,pairId] = pair
            if key not in cand.keys():
                cand[key] = [pairId]
            else:
                cand[key].append(pairId)
        
        # output holder preparing
        keys, scores, bboxes, hdet, odet = [], [], [], [], []
        for i in range(80):
            keys.append([])
            scores.append([])
            bboxes.append([])
            hdet.append([])
            odet.append([])

        # model preparing
        net = TIN_ResNet50()
        net = load_model(net, cfg.WEIGHT_PATH)
        net.testMode = True
        net.eval()
        net = net.cuda()

        # proceed testing
        test_HICO(net, cand, db, args.human_thres, args.object_thres, \
            keys, scores, bboxes, hdet, odet)

        for i in range(80):
            try:
                scores[i] = np.concatenate(scores[i], axis=0) 
            except:
                pass
            
            try:
                bboxes[i]    = np.concatenate(bboxes[i], axis=0)
            except:
                pass
            
            keys[i]   = np.array(keys[i])
            hdet[i]   = np.array(hdet[i])
            odet[i]   = np.array(odet[i])
    
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # map, mrec = get_map(keys, scores, bboxes, hdet, odet)
        map, mrec = get_map(keys, scores, bboxes)
        print(map, mrec)
        
        # pickle.dump({'ap': map, 'rec': mrec}, open(output_dir + '/detail.pkl', 'wb'))
        with open(output_dir + '/result.txt', 'w') as f:
            f.write('total    ap: %.4f rec: %.4f \n' % (float(np.mean(map)), float(np.mean(mrec))))
            f.write('rare     ap: %.4f rec: %.4f \n' % (float(np.mean(map[cnt < 10])),  float(np.mean(mrec[cnt < 10]))))
            f.write('non-rare ap: %.4f rec: %.4f \n' % (float(np.mean(map[cnt >= 10])), float(np.mean(mrec[cnt >= 10]))))