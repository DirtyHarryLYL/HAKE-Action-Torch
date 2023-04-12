import pickle
from re import sub
import numpy as np
import h5py
import os.path as osp
import torch
from torch.utils.data import Dataset
from util.object_to_hoi import obj_range_padding, obj_range_cnt
from util.pasta_utils import P_num


def get_var_rule(rule):
    # rule (600,n,76)
    rule_shape = rule.shape
    var_rule = np.zeros_like(rule).reshape(-1, rule_shape[-1])
    unique_rule = [np.unique(x) for x in rule.reshape(-1, rule_shape[-1])]
    rule_lens = np.array([len(x) for x in unique_rule])
    for i, x in enumerate(unique_rule):
        var_rule[i, :rule_lens[i]] = x

    return var_rule.reshape(rule.shape), rule_lens.reshape(rule.shape[:-1])

def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2

def jitter_boxes(box, width, height):
    """
    Jitter box to simulate proposals: apply random small translation/rescaling
    Small: to keep IoU=0.5 with non jittered box -> could try to compute it, or just experimentally 
    By hand: imagine your object is square (nxn): if you shift box by unit 1 in same direction in both x and y (worst case), \
    then IoU is given by (n-1)*(n-1)/((n-1)*(n-1)+4*(n-1)+2). Setting n=5 provide worst IoU<0.5, setting n=6 gives IoU>0.5
    So we accept deformation 
    """

    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

    # Small translation (x,y)
    xt_min = -np.maximum((x2-x1)/5,1)
    xt_max = np.maximum((x2-x1)/5,1)
    yt_min = -np.maximum((y2-y1)/5,1)
    yt_max = np.maximum(1,(y2-y1)/5)
    x_trans = (xt_max-xt_min)*np.random.random() + xt_min
    y_trans = (yt_max-yt_min)*np.random.random() + yt_min

    # Transform
    x1_new = x1 + x_trans
    x2_new = x2 + x_trans
    y1_new = y1 + y_trans
    y2_new = y2 + y_trans 

    # Apply small rescaling: keep aspect ratio but scale it
    scale_factor = np.random.uniform(pow(2,-1/4), pow(2,1/4)) # value taken from "Learning to segment object candidates"
    center_x = (x1_new+x2_new)/2
    center_y = (y1_new+y2_new)/2
    w_box = (x2_new-x1_new+1)
    h_box = (y2_new-y1_new+1)
    w_box_scale = w_box*scale_factor
    h_box_scale = h_box*scale_factor
    x1_new = center_x - w_box_scale/2
    x2_new = center_x + w_box_scale/2
    y1_new = center_y - h_box_scale/2
    y2_new = center_y + h_box_scale/2  

    # Clip to image
    x1_new,y1_new,x2_new,y2_new = clip_xyxy_to_image(x1_new, y1_new, x2_new, y2_new, height, width)

    # Case where no transformation : x2-x1<1 or y2-y2<1
    if x2_new-x1_new < 1:
        x2_new = x2
        x1_new = x1

    if y2_new-y1_new < 1:
        y2_new = y2
        y1_new = y1

    jittered_box = np.array([x1_new,y1_new,x2_new,y2_new])

    return jittered_box

def flip_horizontal(box, width, height):
    """
    Flip box horinzontally
    """
    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

    x1_flip = width-x2
    x2_flip = width-x1

    flip_box = np.array([x1_flip, y1, x2_flip, y2])

    return flip_box

def load_pair_objects(info, sub_id, obj_id, jittering=False, rescale=True):
    """
    Load object boxes for all candidate pairs
    Input:  im_id
            cand_id : index of candidate pair in image
    Output:
            objects (1,2,6) : [x1,y1,x2,y2,obj_cat,obj_score]
    """
    hbox   = info['boxes'][sub_id, :]
    obox   = info['boxes'][obj_id, :]
    width  = 1
    height = 1
    if jittering:
        width  = info['width']
        height = info['height']
        hbox = jitter_boxes(hbox, width, height)
        obox = jitter_boxes(obox, width, height)

        # Horizontal flip
        if np.random.binomial(1,0.5):
            hbox = flip_horizontal(hbox, width, height)
            obox = flip_horizontal(obox, width, height)
    
    spatial = np.concatenate([hbox, obox])
    
    if rescale:
        width  = info['width']
        height = info['height']
        spatial[[0, 2, 4, 6]] /= width
        spatial[[1, 3, 5, 7]] /= height
    
    return spatial, np.array([width, height])

class HICO_gt_test_set(Dataset):

    def __init__(self, config):
        self.config = config
        self.data_dir = config.TEST.DATA_DIR
        self.split = 'test'
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_gt_' + self.split + '_hake_with_pvp_label_with_labels_r.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_gt.pkl'), 'rb'))
        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        hbox = info['boxes'][sub_id, :]
        obox = info['boxes'][obj_id, :]
        spatial = np.concatenate([hbox, obox])

        labels_r_ = info['labels_r'][cand_id]

        obj_class = info['obj_classes'][obj_id] - 1
        key = int(info['filename'][:-4].split('_')[-1])
        hdet = 1.
        odet = 1.
        if self.config.MODEL.NUM_CLASS == 600:
            if self.config.MODEL.NUM_CLASS_SELECT < 600:
                begin, end = obj_range_padding[obj_class][0] - \
                             1, obj_range_padding[obj_class][1]
            else:
                begin, end = 0, 600

            rule = self.rule[begin:end, :, :]
            gt_range = range(begin, end)

        var_rule, rule_lens = None, None
        gt_obj = np.zeros(80)
        gt_obj[obj_class] = 1
        gt_pvp = np.concatenate([info[key][cand_id] for key in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']])

        return key, spatial, obj_class, gt_pvp, hdet, odet, gt_obj, rule, gt_range, var_rule, rule_lens, labels_r_

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        key, spatial, obj_class, gt_pvp, hdet, odet, gt_obj, rule, gt_range, var_rule, rule_lens, labels_r_ = zip(
            *data)
        gt_pvp = np.array(gt_pvp)
        gt_obj = np.array(gt_obj)
        rule = np.array(rule)
        gt_range = np.array(gt_range)
        var_rule = np.array(var_rule)
        rule_lens = np.array(rule_lens)
        labels_r_ = np.array(labels_r_)

        output = {}

        output['key'] = np.array(key)
        output['spatial'] = np.array(spatial)
        output['obj_class'] = np.array(obj_class)
        output['gt_pvp'] = torch.from_numpy(gt_pvp).float()
        output['gt_obj'] = torch.from_numpy(gt_obj).float()
        output['hdet'] = np.array(hdet)
        output['odet'] = np.array(odet)
        output['rule'] = torch.from_numpy(rule).long()  # 600*76
        output['gt_range'] = torch.from_numpy(gt_range).long()
        output['labels_r_'] = labels_r_

        return output
