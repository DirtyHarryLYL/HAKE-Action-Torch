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
 
from hakeaction.models.DJRN             import DJRN_ResNet50
from hakeaction.core.utils.pointnet     import *
from hakeaction.datasets.HICO_dataset   import HICO_Testset
from hakeaction.configs.config_DJRN     import cfg
from datasets.utils.HICO_DET_utils      import *
from datasets.utils.hico_generate_utils import *

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

    parser.add_argument('--PointnetWeight', 
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

def test_HICO(net, Test_RCNN, human_thres, object_thres, keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg, scores_HO):
    
    category_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, \
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
    
    vertex_choice    = np.array(pickle.load(open(cfg.VERTEX_CHOICE_PATH,'rb'), encoding='latin1'))[:,0]
    smplx_model_data = pickle.load(open(cfg.SMPLX_MODEL_DATA_PATH, 'rb'), encoding='latin1')
    pointNet         = PointNetHico().cuda().eval() # TODO: load pretrained pointNet 
    list_hoi, order_obj_list, obj_para_dict = get_order_obj()

    for key,imgData in tqdm(Test_RCNN.items()):
        
        # data preparing
        image_id = int(key)
        img_path = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
        im       = cv2.imread(img_path)
        im_orig  = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_shape = im_orig.shape
        im_orig  = im_orig.transpose(2, 0, 1)
        image    = torch.from_numpy(im_orig).float().cuda().unsqueeze(0)
        
        This_image     = []
        blobs          = {}
        blobs['image'] = image

        # Note that Human_out and Object are index 
        for Human_out in range(len(imgData)):
            if imgData[Human_out][1] != 'Human':
                continue
            Human_score = imgData[Human_out][5]
            Human_bbox  = imgData[Human_out][2]
            Human_bbox  = np.array([0, Human_bbox[0],  Human_bbox[1],  Human_bbox[2],  Human_bbox[3]]).reshape(1,5)

            if Human_score > human_thres: # This is a valid human
                
                blobs['H_num']      = 0
                blobs['H_boxes']    = [np.empty((0, 5), np.float64)]
                blobs['O_boxes']    = [np.empty((0, 5), np.float64)]
                blobs['sp']         = [np.empty((0, 64, 64, 3), np.float64)]
                blobs['pc_att_map'] = np.expand_dims(pickle.load(open(cfg.DATA_DIR + '/att_map.pkl', 'rb')), axis=0)
                blobs['att_2D_map'] = []
                blobs['sp']         = []
                blobs['smplx']      = []
                blobs['pc']         = []
                objectIndex         = []

                for Object in range(len(imgData)):
                    if Human_out == Object:
                        continue
                    
                    Object_score = imgData[Object][5]
                    Object_bbox  = imgData[Object][2]
                    Object_bbox  = np.array([0, Object_bbox[0],  Object_bbox[1],  Object_bbox[2],  Object_bbox[3]]).reshape(1,5)
                    
                    if Object_score > object_thres: # This is a valid object

                        tmp_sp, tmp_att_2D_map = Get_next_sp_with_pose_calAttMap(  
                                                    imgData[Human_out][2], imgData[Object][2], imgData[Human_out][6])
                        blobs['sp'].append(tmp_sp.reshape((1, 64, 64, 3)))
                        blobs['att_2D_map'].append(tmp_att_2D_map)
                        blobs['H_boxes'].append(Human_bbox)
                        blobs['O_boxes'].append(Object_bbox)
                        blobs['H_num'] += 1
                        objectIndex.append(Object)

                        if (imgData[Object][-1] != None) and os.path.exists(cfg.SMPLX_TEST_PATH + '/results/HICO_test2015_%08d/%03d.pkl' % (image_id, imgData[Object][-1]['id'])):
                            result = pickle.load(open(cfg.SMPLX_TEST_PATH + '/results/HICO_test2015_%08d/%03d.pkl' % (image_id, imgData[Object][-1]['id']),  "rb" ), encoding="bytes")
                            blobs['smplx'].append(np.concatenate([
                                        result['left_hand_pose'], result['right_hand_pose'],
                                        result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                        result['expression'], result['betas'],
                                    ], axis=1))
                            
                            obj_name = order_obj_list[imgData[Object][4]]
                            tmp_pc   = getPointCloudFeature(pointNet=pointNet,             cfg=cfg,                      obj_name=obj_name,
                                                            hbox=imgData[Human_out][2],    obox=imgData[Object][2],      imgPath=img_path,     key=image_id,
                                                            idx=imgData[Object][-1]['id'], model_data=smplx_model_data,  smplx_result=result,
                                                            vertex_choice=vertex_choice,   obj_para_dict=obj_para_dict,  smplx_path=cfg.SMPLX_TEST_PATH)
                            blobs['pc'].append(tmp_pc)
                        else:
                            blobs['smplx'].append(np.zeros((1, 85)))
                            blobs['pc'].append(np.zeros((1, 1228, 256)))
                
                if blobs['H_num'] == 0:
                    continue

                blobs['H_boxes']    = np.concatenate(blobs['H_boxes'],    axis=0)
                blobs['O_boxes']    = np.concatenate(blobs['O_boxes'],    axis=0)
                blobs['sp']         = np.concatenate(blobs['sp'],         axis=0)
                blobs['att_2D_map'] = np.expand_dims(np.concatenate(blobs['att_2D_map'], axis=0), axis=0)
                blobs['smplx']      = np.expand_dims(np.concatenate(blobs['smplx'],      axis=0), axis=0)
                blobs['pc']         = np.expand_dims(np.concatenate(blobs['pc'],         axis=0), axis=0)

                # move data to device(GPU)
                for k, v in blobs.items():
                    if k in ['image_id', 'H_num', 'image']:
                        continue
                    # print(k)
                    blobs[k] = torch.from_numpy(v).float().cuda()

                # inference, test forward 
                cls_prob_HO, cls_prob_H, cls_prob_O, cls_prob_sp = net(blobs)
                cls_prob_HO     = cls_prob_HO.cpu().detach().numpy()
                cls_prob_H      = cls_prob_H.cpu().detach().numpy()
                cls_prob_O      = cls_prob_O.cpu().detach().numpy()
                cls_prob_sp     = cls_prob_sp.cpu().detach().numpy()

                for i in range(blobs['H_num']):
                    Object  = objectIndex[i]
                    Object_score = imgData[Object][5]
                    classid = imgData[Object][4] - 1

                    keys[classid].append(image_id)
                    scores_H[classid].append(
                            cls_prob_H[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_O[classid].append(
                            cls_prob_O[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_sp[classid].append(
                            cls_prob_sp[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    
                    hdet[classid].append(Human_score)
                    odet[classid].append(Object_score)
                    
                    hbox = np.array(imgData[Human_out][2]).reshape(1, -1) 
                    obox = np.array(imgData[Object][2]).reshape(1, -1)
                    bboxes[classid].append(np.concatenate([hbox, obox], axis=1))     

                    scores_HO[classid].append(  
                        # with lis
                        # cls_prob_HO[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                        # getSigmoid(9, 1, 3, 0, Human_out[5]) * \
                        # getSigmoid(9, 1, 3, 0, Object[5])
                        
                        # without lis
                        cls_prob_HO[i][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                        Human_score * \
                        Object_score
                        )

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    cfg.WEIGHT_PATH  = args.weight
    args.model = args.weight.split('/')[-1]
    np.random.seed(cfg.RNG_SEED)

    # prepare output folder
    output_dir  = cfg.ROOT_DIR + '/results/Test_3Ddata_' + args.weight.split('/')[-1].strip('.tar') + '_Htre-{}_Otre-{}'.format(args.human_thres, args.object_thres)
    print("==> [test HICO] output_dir: ", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.stage == 1:
        # data loading
        Test_RCNN = pickle.load( open( '/Disk3/zhanke/data/pami/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_smplx.pkl', "rb" ), encoding="bytes")

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
        # net = load_model(net, cfg.WEIGHT_PATH) # TODO
        net.testMode = True
        net.eval()
        net = net.cuda()

        # proceed testing
        test_HICO(net, Test_RCNN, args.human_thres, args.object_thres, \
            keys, scores_H, scores_O, scores_sp, bboxes, hdet, odet, pos, neg, scores_HO)

        # save test results
        # print(scores_H)
        for i in range(80):
            
            try:
                scores_H[i]  = np.concatenate(scores_H[i], axis=0)
            except:
                pass
            
            try:
                scores_O[i]  = np.concatenate(scores_O[i], axis=0)
            except:
                pass
            
            try:
                scores_sp[i] = np.concatenate(scores_sp[i], axis=0)
            except:
                pass
            
            try:
                scores_HO[i] = np.concatenate(scores_HO[i], axis=0) 
            except:
                pass
            
            try:
                bboxes[i]    = np.concatenate(bboxes[i], axis=0)
            except:
                pass
            
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
    
    mAP, mrec = get_map(keys, scores_HO, bboxes)
    pickle.dump(mAP,   open(os.path.join(output_dir,  'map.pkl'), 'wb'))
    pickle.dump(mrec,  open(os.path.join(output_dir, 'mrec.pkl'), 'wb'))

    MmAP, Mmrec = np.mean(mAP), np.mean(mrec)
    print("\n==> Evaluation Result without NIS: mAP={}, mrec={}".format(MmAP, Mmrec))
    exit()

    NISThreshold = 0.9
    mAP, mrec = get_map_with_NIS(keys, scores_HO, bboxes, pos, neg, NISThreshold)
    mAP, mrec = np.mean(mAP), np.mean(mrec)
    print("==> Evaluation  Result  with  NIS: mAP={}, mrec={}".format(mAP, mrec))