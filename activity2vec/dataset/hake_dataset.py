#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
import torch
import torch.utils
import cv2
import numpy as np
import random
import os
import sys
import inspect
import pickle
import base64
import lmdb
import os.path as osp
import imageio
import json
import copy
from easydict import EasyDict as edict
from tqdm import tqdm
from turbojpeg import TurboJPEG
reader = TurboJPEG()

# Transform between object and string items.
def obj2str(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()

def str2obj(strr):
    return pickle.loads(base64.b64decode(strr))

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def im_read(im_path):
    try:
        im = reader.decode(open(im_path, 'rb').read(), 1)
    except:
        im = cv2.imread(im_path)
    if im is None:
        im = imageio.imread(im_path)
        if im.shape[-1] == 4:
            im = rgba2rgb(im)
            im = im[:,:,::-1]
            im = np.array(im)
        else:
            raise NotImplementedError
    return im

# Generate the skeleton input for each human pose.
def draw_relation(joints, size=64, is_fake=False):
    if is_fake:
        return np.zeros((size, size), dtype='float32')
    else:
        joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10], [11, 17], [12, 17], [11, 13], [12, 14], [13, 15], [14, 16]]
        color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        skeleton = np.zeros((size, size, 1), dtype='float32')
        draw_joints = np.zeros((18, 2), dtype='float32')
        draw_joints[0:17] = joints
        draw_joints[17] = (draw_joints[5] + draw_joints[6]) / 2
        for i in range(len(joint_relation)):
            cv2.line(skeleton, (int(draw_joints[joint_relation[i][0]][0]*size), int(draw_joints[joint_relation[i][0]][1]*size)), (int(draw_joints[joint_relation[i][1]][0]*size), int(draw_joints[joint_relation[i][1]][1]*size)), color[i])
        skeleton = skeleton[:, :, 0]
        return skeleton

class hake_train(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(hake_train, self).__init__()
        self.cfg = cfg
        self.db = lmdb.open(self.cfg.DATA.ANNO_DB_PATH)
        self.txn_db = self.db.begin(write=False)
        self.image_folder_list = json.load(open(self.cfg.DATA.IMAGE_FOLDER_LIST,'r'))
        self.image_list = [key for key, _ in self.txn_db.cursor() if key.decode().split('/')[0] in self.cfg.TRAIN.DATA_SPLITS]
        self.visualize = False
        # the numbers of positive and negative samples.
        self.pos_num = int(self.cfg.TRAIN.HUMAN_PER_IM * self.cfg.TRAIN.POS_RATIO)
        self.neg_num = self.cfg.TRAIN.HUMAN_PER_IM - self.pos_num

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        current_key_raw   = self.image_list[idx]
        image_id          = current_key_raw.decode()
        current_data      = str2obj(self.txn_db.get(current_key_raw))
        dataset, filename = image_id.split('/')
        
        # Load image and normalize.
        im_path  = osp.join(self.cfg.DATA.DATA_DIR, self.image_folder_list[dataset], filename)
        im       = im_read(im_path)
        ori_im_shape = im.shape
        if not self.visualize:
            im   = im.astype(np.float32, copy=True)
            im  -= np.array(self.cfg.PIXEL_MEANS)
        else:
            print('[Warning] Visualization Mode!')
        im_shape = im.shape

        # Resize the image of each idx.
        # long side to 512.
        ratio = 1.0
        if im_shape[0] > im_shape[1]:
            if im_shape[0] > 512:
                ratio = 512. / im_shape[0]
        else:
            if im_shape[1] > 512:
                ratio = 512. / im_shape[1]
        
        im       = cv2.resize(im, (int(ratio*im_shape[1]), int(ratio*im_shape[0])))
        im_shape = im.shape
        image    = im.transpose(2, 0, 1)

        # Collect the positive and negative annotations.
        gt_annos, neg_annos = [], []
        for anno in current_data:
            if anno.gt_flag:
                gt_annos.append(anno)
            else:
                neg_annos.append(anno)
        
        gt_anno_length, neg_anno_length = len(gt_annos), len(neg_annos)

        # Augment the positive and negative data.
        if self.pos_num > gt_anno_length:
            gt_anno_idxs = np.concatenate((np.arange(gt_anno_length), np.random.choice(gt_anno_length, self.pos_num-gt_anno_length)))
        else:
            gt_anno_idxs = np.arange(gt_anno_length)
        
        if self.neg_num > neg_anno_length:
            neg_anno_idxs = np.arange(neg_anno_length)
        else:
            neg_anno_idxs = np.random.permutation(neg_anno_length)[:self.neg_num]
        
        gt_num, neg_num = len(gt_anno_idxs), len(neg_anno_idxs)
        anno_num = gt_num + neg_num
        
        annos = edict()
        annos.gt_flag = np.zeros((anno_num, ), dtype=int)
        annos.gt_flag[:gt_num] = 1
        annos.verbs = np.zeros((anno_num, self.cfg.DATA.NUM_VERBS), dtype=np.float32)
        annos.human_bboxes = np.zeros((anno_num, 4), dtype=np.float32)
        annos.part_bboxes = np.zeros((anno_num, self.cfg.DATA.NUM_PARTS, 4), dtype=np.float32)
        annos.pasta = edict()
        annos.pasta.foot = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.FOOT), dtype=np.float32)
        annos.pasta.leg = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.LEG), dtype=np.float32)
        annos.pasta.hip = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.HIP), dtype=np.float32)
        annos.pasta.hand = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.HAND), dtype=np.float32)
        annos.pasta.arm = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.ARM), dtype=np.float32)
        annos.pasta.head = np.zeros((anno_num, self.cfg.DATA.NUM_PASTAS.HEAD), dtype=np.float32)
        annos.pasta.binary = np.zeros((anno_num, self.cfg.DATA.NUM_PARTS), dtype=np.float32)
        annos.skeletons = np.zeros((anno_num, 1, self.cfg.DATA.SKELETON_SIZE, self.cfg.DATA.SKELETON_SIZE), dtype=np.float32)

        # Load the annotations in one batch.
        for gt_anno_aug_idx, gt_anno_ori_idx in enumerate(gt_anno_idxs):
            this_anno = gt_annos[gt_anno_ori_idx]
            global_idx = gt_anno_aug_idx

            if this_anno.keypoints is not None:
                keypoints = this_anno.keypoints
                height, width, _ = ori_im_shape
                keypoints = np.array(keypoints).reshape(17, 3)
                keypoints[:, 0] /= width
                keypoints[:, 1] /= height
                skeleton_image = draw_relation(keypoints[:, :2])
                annos.skeletons[global_idx, 0] = skeleton_image
            else:
                skeleton_image = draw_relation(None, is_fake=True)
                annos.skeletons[global_idx, 0] = skeleton_image

            if len(this_anno.verbs) > 0:
                annos.verbs[global_idx][np.array(this_anno.verbs)] = 1
            else:
                annos.verbs[global_idx][57] = 1
            annos.human_bboxes[global_idx] = np.array(this_anno.human_bbox)
            if this_anno.part_bboxes is None:
                annos.part_bboxes[global_idx] = np.tile(this_anno.human_bbox, [10, 1])
            else:
                annos.part_bboxes[global_idx] = np.array(this_anno.part_bboxes[:, 1:])
            annos.pasta.foot[global_idx][np.array(this_anno.pasta.foot)] = 1
            annos.pasta.leg[global_idx][np.array(this_anno.pasta.leg)] = 1
            annos.pasta.hip[global_idx][np.array(this_anno.pasta.hip)] = 1
            annos.pasta.hand[global_idx][np.array(this_anno.pasta.hand)] = 1
            annos.pasta.arm[global_idx][np.array(this_anno.pasta.arm)] = 1
            annos.pasta.head[global_idx][np.array(this_anno.pasta.head)] = 1
            if len(this_anno.pasta.binary) > 0:
                annos.pasta.binary[global_idx][np.array(this_anno.pasta.binary)] = 1

        for neg_anno_aug_idx, neg_anno_ori_idx in enumerate(neg_anno_idxs):
            this_anno = neg_annos[neg_anno_ori_idx]
            global_idx = neg_anno_aug_idx + gt_num

            if this_anno.keypoints is not None:
                keypoints = this_anno.keypoints
                height, width, _ = ori_im_shape
                keypoints = np.array(keypoints).astype(np.float32).reshape(17, 3)
                keypoints[:, 0] /= width
                keypoints[:, 1] /= height
                skeleton_image = draw_relation(keypoints[:, :2])
                annos.skeletons[global_idx, 0] = skeleton_image
            else:
                skeleton_image = draw_relation(None, is_fake=True)
                annos.skeletons[global_idx, 0] = skeleton_image

            annos.verbs[global_idx][57] = 1
            annos.human_bboxes[global_idx] = np.array(this_anno.human_bbox)
            if this_anno.part_bboxes is None:
                annos.part_bboxes[global_idx] = np.tile(this_anno.human_bbox, [10, 1])
            else:
                annos.part_bboxes[global_idx] = np.array(this_anno.part_bboxes[:, 1:])
            annos.pasta.foot[global_idx][-1] = 1
            annos.pasta.leg[global_idx][-1] = 1
            annos.pasta.hip[global_idx][-1] = 1
            annos.pasta.hand[global_idx][-1] = 1
            annos.pasta.arm[global_idx][-1] = 1
            annos.pasta.head[global_idx][-1] = 1

        annos.human_bboxes    *= ratio
        annos.part_bboxes     *= ratio

        annos.part_bboxes[:, :, 0] = np.maximum(annos.part_bboxes[:, :, 0], 0)
        annos.part_bboxes[:, :, 1] = np.maximum(annos.part_bboxes[:, :, 1], 0)
        annos.part_bboxes[:, :, 2] = np.minimum(annos.part_bboxes[:, :, 2], im_shape[1])
        annos.part_bboxes[:, :, 3] = np.minimum(annos.part_bboxes[:, :, 3], im_shape[0])

        return image, annos


class hake_test(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(hake_test, self).__init__()
        self.cfg = cfg
        self.db = lmdb.open(self.cfg.DATA.PRED_DB_PATH)
        self.txn_db = self.db.begin(write=False)
        self.image_folder_list = json.load(open(self.cfg.DATA.IMAGE_FOLDER_LIST,'r'))
        self.image_list = [key for key, _ in self.txn_db.cursor()]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        current_key_raw   = self.image_list[idx]
        image_id          = current_key_raw.decode()
        current_data      = str2obj(self.txn_db.get(current_key_raw))
        dataset, filename = image_id.split('/')

        # Load image and normalize.
        im_path  = osp.join(self.cfg.DATA.DATA_DIR, self.image_folder_list[dataset], filename)
        image    = im_read(im_path)
        ori_im_shape = image.shape
        image    = image.astype(np.float32, copy=True)
        image   -= self.cfg.PIXEL_MEANS
        im_shape = image.shape
        image    = image.transpose(2, 0, 1)
        
        annos     = edict()
        anno_num  = len(current_data)
        annos.human_bboxes = np.zeros((anno_num, 4), dtype=np.float32)
        annos.part_bboxes = np.zeros((anno_num, self.cfg.DATA.NUM_PARTS, 4), dtype=np.float32)
        annos.human_scores = np.zeros((anno_num, ), dtype=np.float32)
        annos.skeletons = np.zeros((anno_num, 1, self.cfg.DATA.SKELETON_SIZE, self.cfg.DATA.SKELETON_SIZE), dtype=np.float32)

        # Load the predicted annotations in one batch.
        for anno_idx, anno in enumerate(current_data):
            if anno.keypoints is not None:
                keypoints = anno.keypoints
                height, width, _ = ori_im_shape
                keypoints = np.array(keypoints).astype(np.float32).reshape(17, 3)
                keypoints[:, 0] /= width
                keypoints[:, 1] /= height
                skeleton_image = draw_relation(keypoints[:, :2])
                annos.skeletons[anno_idx, 0] = skeleton_image
            else:
                skeleton_image = draw_relation(None, is_fake=True)
                annos.skeletons[anno_idx, 0] = skeleton_image
            annos.human_bboxes[anno_idx] = np.array(anno.human_bbox)
            annos.part_bboxes[anno_idx] = anno.part_bboxes
            annos.human_scores[anno_idx] = anno.human_score
        annos.part_bboxes[:, :, 0] = np.maximum(annos.part_bboxes[:, :, 0], 0)
        annos.part_bboxes[:, :, 1] = np.maximum(annos.part_bboxes[:, :, 1], 0)
        annos.part_bboxes[:, :, 2] = np.minimum(annos.part_bboxes[:, :, 2], im_shape[1])
        annos.part_bboxes[:, :, 3] = np.minimum(annos.part_bboxes[:, :, 3], im_shape[0])
        
        return image, annos, image_id