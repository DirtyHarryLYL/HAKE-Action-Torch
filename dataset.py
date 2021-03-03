import pickle
import numpy as np
import h5py
import os.path as osp
import torch
from torch.utils.data import Dataset
from HICO_DET_utils import obj_range
from object_to_hoi import obj_range_padding, obj_range_cnt

P_num = {
    'P0': 12,
    'P1': 10,
    'P2': 5,
    'P3': 31,
    'P4': 5,
    'P5': 13,
    'labels_r': 600,
    'labels_v': 29,  # V-COCO
    'labels_a': 80  # ava
}


def trans_6v_to_10v(x):
    mapping = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
    y = [x[mapping[i]] for i in range(10)]
    return y


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

    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    # Small translation (x,y)
    xt_min = -np.maximum((x2 - x1) / 5, 1)
    xt_max = np.maximum((x2 - x1) / 5, 1)
    yt_min = -np.maximum((y2 - y1) / 5, 1)
    yt_max = np.maximum(1, (y2 - y1) / 5)
    x_trans = (xt_max - xt_min) * np.random.random() + xt_min
    y_trans = (yt_max - yt_min) * np.random.random() + yt_min

    # Transform
    x1_new = x1 + x_trans
    x2_new = x2 + x_trans
    y1_new = y1 + y_trans
    y2_new = y2 + y_trans

    # Apply small rescaling: keep aspect ratio but scale it
    scale_factor = np.random.uniform(pow(2, -1 / 4),
                                     pow(2, 1 / 4))  # value taken from "Learning to segment object candidates"
    center_x = (x1_new + x2_new) / 2
    center_y = (y1_new + y2_new) / 2
    w_box = (x2_new - x1_new + 1)
    h_box = (y2_new - y1_new + 1)
    w_box_scale = w_box * scale_factor
    h_box_scale = h_box * scale_factor
    x1_new = center_x - w_box_scale / 2
    x2_new = center_x + w_box_scale / 2
    y1_new = center_y - h_box_scale / 2
    y2_new = center_y + h_box_scale / 2

    # Clip to image
    x1_new, y1_new, x2_new, y2_new = clip_xyxy_to_image(x1_new, y1_new, x2_new, y2_new, height, width)

    # Case where no transformation : x2-x1<1 or y2-y2<1
    if x2_new - x1_new < 1:
        x2_new = x2
        x1_new = x1

    if y2_new - y1_new < 1:
        y2_new = y2
        y1_new = y1

    jittered_box = np.array([x1_new, y1_new, x2_new, y2_new])

    return jittered_box


def flip_horizontal(box, width, height):
    """
    Flip box horinzontally
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    x1_flip = width - x2
    x2_flip = width - x1

    flip_box = np.array([x1_flip, y1, x2_flip, y2])

    return flip_box


def load_pair_objects(info, sub_id, obj_id, jittering=False, rescale=False):
    """
    Load object boxes for all candidate pairs
    Input:  im_id
            cand_id : index of candidate pair in image
    Output:
            objects (1,2,6) : [x1,y1,x2,y2,obj_cat,obj_score]
    """
    hbox = info['boxes'][sub_id, :]
    obox = info['boxes'][obj_id, :]
    width = info['width']
    height = info['height']
    if jittering:
        hbox = jitter_boxes(hbox, width, height)
        obox = jitter_boxes(obox, width, height)

        # Horizontal flip
        if np.random.binomial(1, 0.5):
            hbox = flip_horizontal(hbox, width, height)
            obox = flip_horizontal(obox, width, height)

    spatial = np.concatenate([hbox, obox])

    if rescale:
        spatial[[0, 2, 4, 6]] /= width
        spatial[[1, 3, 5, 7]] /= height

    return spatial, np.array([width, height])


class HICO_test_set(Dataset):

    def __init__(self, config, split='test'):

        self.config = config
        self.data_dir = config.TEST.DATA_DIR
        self.split = split
        # self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake.pkl'), 'rb'))
        # self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '.pkl'), 'rb'))
        if not config.TEST.get('EVAL_ALL') or not self.config.TEST.EVAL_ALL:
            self.db = pickle.load(
                open(osp.join(self.data_dir, 'db_' + self.split + '_hake_H_0.8_O_0.3_with_pvp_label.pkl'), 'rb'))
            self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_H_0.8_O_0.3.pkl'), 'rb'))
        else:
            print('test all.')
            self.db = pickle.load(
                open(osp.join(self.data_dir, 'db_' + self.split + '_hake_with_pvp_label.pkl'), 'rb'))
            self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '.pkl'), 'rb'))

        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys[:-1]

        rules = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # 600*n*6
        if isinstance(rules, dict):
            self.rule = rules['rules_index']
            self.rule_lens = rules['rules_len']
        else:
            self.rule = np.array(rules)

        # print(self.rule)

    def __len__(self):
        return self.cand.shape[0]

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        spatial, shape = load_pair_objects(info, sub_id, obj_id, rescale=False)
        feature, label = [], []

        for i in self.F_keys:
            feature.append([])

        with h5py.File(osp.join(self.data_dir, 'hake', self.split, str(im_id) + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key == 'FH':
                    feature[i].append(f[key][sub_id, ...])
                elif key == 'FO':
                    feature[i].append(f[key][obj_id, ...])
                elif key == 'FS':
                    feature[i].append(f[key][0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][cand_id, ...])

        gt_parts = np.zeros(6)
        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label.append(info[key][cand_id])
            gt_parts[i] = np.max(label[-1][:-1])

        gt_parts = trans_6v_to_10v(gt_parts)

        begin, end = obj_range_padding[info['obj_classes'][obj_id] - 1][0] - 1, \
                     obj_range_padding[info['obj_classes'][obj_id] - 1][1]

        rules = self.rule[begin:end]
        rule_lens = None
        if hasattr(self, 'rule_lens'):
            rule_lens = self.rule_lens[begin:end]

        for i in range(len(self.F_keys)):
            feature[i] = np.array(feature[i])

        for i in range(len(self.L_keys)):
            label[i] = np.array(label[i])

        obj_class = info['obj_classes'][obj_id]
        key = int(info['filename'][-10:-4])
        hdet = info['obj_scores'][sub_id]
        odet = info['obj_scores'][obj_id]

        gt_obj = np.zeros(80)
        gt_obj[obj_class - 1] = 1.

        gt_range = range(begin, end)

        return [key, spatial, hdet, odet, obj_class - 1, gt_obj, gt_parts, rules, gt_range, rule_lens] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}
        output['key'] = np.array(blob[0])
        output['spatial'] = np.array(blob[1])
        output['hdet'] = np.array(blob[2])
        output['odet'] = np.array(blob[3])
        output['obj_class'] = np.array(blob[4])
        output['gt_obj'] = torch.from_numpy(np.array(blob[5])).float()
        output['gt_part'] = torch.from_numpy(np.array(blob[6])).float()
        output['rule'] = torch.from_numpy(np.array(blob[7])).long()
        output['gt_range'] = torch.from_numpy(np.array(blob[8])).long()

        if hasattr(self, 'rule_lens'):
            output['rule_lens'] = torch.from_numpy(np.array(blob[9])).long()

        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 10])).float()

        for i in range(len(self.L_keys)):
            output[self.L_keys[i]] = torch.from_numpy(np.array(blob[i + 10 + len(self.F_keys)])).float()

        return output


class HICO_train_set(Dataset):

    def __init__(self, config, split='trainval', train_mode=True):

        self.config = config

        self.data_dir = config.TRAIN.DATA_DIR
        self.sampler_name = config.TRAIN.DATASET.SAMPLER_NAME
        self.num_neg = config.TRAIN.DATASET.NUM_NEG
        self.jittering = config.TRAIN.DATASET.JITTERING

        self.split = split
        self.train_mode = train_mode

        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake.pkl'), 'rb'))
        self.cand_pos = pickle.load(open(osp.join(self.data_dir, 'cand_positive_' + self.split + '.pkl'), 'rb'))
        self.cand_neg = pickle.load(open(osp.join(self.data_dir, 'cand_negative_' + self.split + '.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys

        rules = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # 600*n*6
        if isinstance(rules, dict):
            self.rule = rules['rules_index']
            self.rule_lens = rules['rules_len']
        else:
            self.rule = np.array(rules)

        cand_cat = self.cand_neg[:, 3]
        self.idx_match_object_candneg = {}
        for obj_cat in range(1, 81):
            self.idx_match_object_candneg[obj_cat] = np.where(cand_cat == obj_cat)[0]

        cand_key = self.cand_neg[:, 0]
        self.idx_match_image_candneg = {}
        for i in range(len(cand_key)):
            if cand_key[i] not in self.idx_match_image_candneg:
                self.idx_match_image_candneg[cand_key[i]] = []
            self.idx_match_image_candneg[cand_key[i]].append(i)

        for key in self.db.keys():
            if key in self.idx_match_image_candneg:
                self.idx_match_image_candneg[key] = np.array(self.idx_match_image_candneg[key])
            self.db[key]['H_mapping'], cnt = [], 0
            for i in range(self.db[key]['obj_classes'].shape[0]):
                if self.db[key]['obj_classes'][i] == 1:
                    self.db[key]['H_mapping'].append(cnt)
                    cnt += 1
                else:
                    self.db[key]['H_mapping'].append(-1)

    def __len__(self):
        return self.cand_pos.shape[0]

    def sample_random_negatives(self, idx, num_to_sample):
        idx_sample = np.random.randint(0, len(idx), size=num_to_sample)
        idx_values = idx[idx_sample]
        return idx_values

    def __getitem__(self, idx):
        """ Intialize input matrices to empty """
        im_id, cand_id, _, _ = self.cand_pos[idx]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]
        # cand_info, spatial, shape, obj_classes, gt_parts, rules = [], [], [], [], [], []
        cand_info, gt_parts, rules, rule_lens, gt_objs, gt_ranges = [], [], [], [], [], []

        sub = info['obj_classes'][sub_id]
        obj = info['obj_classes'][obj_id]
        feature, label = [], []
        for i in self.F_keys:
            feature.append([])
        for i in self.L_keys:
            label.append([])

        cand_info.append(np.array([[im_id, cand_id, sub, obj]]))

        # Sample negatives in the batch (can use different strategies specified by sampler_name)
        if self.sampler_name == 'priority_object':
            """
            This strategy: sample negatives involving the same object category in other images
            """

            # If possible sample negatives from this image, with this object and other human
            num_neg_sampled = 0

            if num_neg_sampled < self.num_neg and im_id in self.idx_match_image_candneg:
                idx_cand_match_image = self.idx_match_image_candneg[im_id]
                if len(idx_cand_match_image) > 0:
                    idx_pair_match_image = list(self.cand_neg[idx_cand_match_image, 1])
                    obj_pair_match_image = info['pair_ids'][idx_pair_match_image, 1]
                    idx_pair_match_image_obj = np.where(obj_pair_match_image == obj_id)[0]
                    idx_cand_match_image_obj = idx_cand_match_image[idx_pair_match_image_obj]
                    idx_neg = self.sample_random_negatives(idx_cand_match_image_obj, min(len(idx_cand_match_image_obj),
                                                                                         self.num_neg - num_neg_sampled))
                    num_neg_sampled += len(idx_neg)
                    cand_info.append(self.cand_neg[idx_neg])

            # Sample additional negatives involving the same object category in other images
            if num_neg_sampled < self.num_neg:
                idx_match_object = self.idx_match_object_candneg[obj]

                if len(idx_match_object) > 0:
                    idx_neg = self.sample_random_negatives(idx_match_object, \
                                                           min(len(idx_match_object), self.num_neg - num_neg_sampled))
                    num_neg_sampled += len(idx_neg)
                    cand_info.append(self.cand_neg[idx_neg])

            # Sample additional negatives randomly from other images (not necessarily with same object category)
            if num_neg_sampled < self.num_neg:
                idx_neg = self.sample_random_negatives(np.arange(len(self.cand_neg)), \
                                                       self.num_neg - num_neg_sampled)
                num_neg_sampled += len(idx_neg)
                cand_info.append(self.cand_neg[idx_neg])

        elif self.sampler_name == 'random':
            """
            Sample negatives totally at random
            """
            num_neg_sampled = 0

            # Sample additional negatives randomly from other images
            if num_neg_sampled < self.num_neg:
                idx_neg = np.random.choice(np.arange(len(self.cand_neg)), \
                                           self.num_neg - num_neg_sampled, replace=False)  # sample in all candidates
                num_neg_sampled += len(idx_neg)
                cand_info.append(self.cand_neg[idx_neg])

        """ Fill the batch """

        cand_info = np.concatenate(cand_info, axis=0)

        for j in range(cand_info.shape[0]):
            im_id, cand_id, sub, obj = cand_info[j, :]
            info = self.db[im_id]
            sub_id, obj_id = info['pair_ids'][cand_id]

            # layout, hw, bbox = load_pair_objects(info, sub_id, obj_id, self.jittering)
            # spatial.append(layout)
            # shape.append(hw)
            # obj_classes.append(info['obj_classes'][obj_id])  # 1-80

            gt_obj = np.zeros(80)  # one-hot object
            gt_obj[info['obj_classes'][obj_id] - 1] = 1.
            gt_objs.append(gt_obj)

            with h5py.File(osp.join(self.data_dir, 'hake', self.split, str(im_id) + '.h5'), 'r') as f:
                for i in range(len(self.F_keys)):
                    key = self.F_keys[i]
                    if key == 'FH':
                        # feature[i].append(f[key][cand_id // 2, ...])
                        feature[i].append(f[key][cand_id, ...])
                    elif key == 'FO':
                        # feature[i].append(f[key][cand_id // 2 + 1, ...])
                        feature[i].append(f[key][cand_id, ...])
                        # feature[i].append(f[key][obj_id, ...])

                    elif key == 'FS':
                        feature[i].append(f[key][0, ...])
                    else:
                        if key[-1] == 'l':
                            feature[i].append(f[key[:3]][cand_id, ...][:1024])
                        elif key[-1] == 'r':
                            feature[i].append(f[key[:3]][cand_id, ...][1024:])
                        else:
                            feature[i].append(f[key][cand_id, ...])

            gt_parts.append(np.zeros(6))
            for i in range(len(self.L_keys)):
                key = self.L_keys[i]
                label[i].append(np.zeros(P_num[key]))
                if key in info and cand_id < len(info[key]) - 1:
                    label[i][-1][info[key][cand_id]] = 1.
                else:
                    if key in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
                        label[i][-1][-1] = 1.
                    else:
                        label[i][-1][obj_range[info['obj_classes'][obj_id] - 1][1] - 1] = 1.

                if key in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
                    gt_parts[-1][i] = np.max(label[i][-1][:-1])
                else:
                    begin, end = obj_range_padding[info['obj_classes'][obj_id] - 1][0] - 1, \
                                 obj_range_padding[info['obj_classes'][obj_id] - 1][1]

                    rules.append(self.rule[begin:end])
                    gt_ranges.append(range(begin, end))
                    if hasattr(self, 'rule_lens'):
                        rule_lens.append(self.rule_lens[begin:end])
                    if self.config.MODE != 'Linear_10v':
                        label[i][-1] = label[i][-1][begin:end]

            gt_parts[-1] = trans_6v_to_10v(gt_parts[-1])

        for i in range(len(self.F_keys)):
            feature[i] = np.array(feature[i])
        for i in range(len(self.L_keys)):
            label[i] = np.array(label[i])

        return [rules, gt_parts, gt_objs, gt_ranges, rule_lens] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))
        output = {}

        output['rule'] = torch.from_numpy(np.concatenate(blob[0])).long()
        output['gt_part'] = torch.from_numpy(np.concatenate(blob[1])).float()
        output['gt_obj'] = torch.from_numpy(np.concatenate(blob[2])).float()
        output['gt_range'] = torch.from_numpy(np.concatenate(blob[3])).long()
        if hasattr(self, 'rule_lens'):
            output['rule_lens'] = torch.from_numpy(np.concatenate(blob[4])).long()

        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 5])).float()

        for i in range(len(self.L_keys)):
            output[self.L_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 5 + len(self.F_keys)])).float()

        return output


class ambi_test_set(Dataset):

    def __init__(self, config, split='test'):

        self.data_dir = config.TRAIN.DATA_DIR
        self.split = split
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake_ambi.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_ambi.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys
        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        spatial, shape = load_pair_objects(info, sub_id, obj_id, rescale=False)
        feature, label = [], []
        with h5py.File(osp.join(self.data_dir, 'hake_ambi', str(im_id) + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key == 'FO':
                    feature.append(f[key][obj_id, ...])
                elif key == 'FR':
                    feature.append(f[key][cand_id, ...])
                elif key == 'FS':
                    feature.append(f[key][0, ...])
                else:
                    if key[-1] == 'l':
                        feature.append(f[key[:3]][sub_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature.append(f[key[:3]][sub_id, ...][1024:])
                    else:
                        feature.append(f[key][sub_id, ...])
                    # feature.append(f[key][sub_id, ...])
        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label.append(np.zeros(P_num[key]))
            if key in info:
                label[-1][info[key][cand_id]] = 1.

        obj_class = info['obj_classes'][obj_id] - 1
        key = info['filename']
        hdet = info['obj_scores'][sub_id]
        odet = info['obj_scores'][obj_id]

        gt_obj = np.zeros(80)
        gt_obj[obj_class] = 1.

        begin, end = obj_range_padding[obj_class][0] - \
                     1, obj_range_padding[obj_class][1]
        rule = self.rule[begin:end, ...]

        gt_range = range(begin, end)

        return [key, spatial, shape, hdet, odet, obj_class, gt_obj, rule, gt_range] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}
        output['key'] = list(blob[0])
        output['spatial'] = torch.from_numpy(np.array(blob[1])).float()
        output['shape'] = torch.from_numpy(np.array(blob[2])).float()
        output['hdet'] = np.array(blob[3])
        output['odet'] = np.array(blob[4])
        output['obj_class'] = np.array(blob[5])
        output['gt_obj'] = torch.from_numpy(np.array(blob[6])).float()
        output['rule'] = torch.from_numpy(np.array(blob[7])).long()
        output['gt_range'] = torch.from_numpy(np.array(blob[8])).long()

        for i in range(len(self.F_keys)):
            k = self.F_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9])).float()
        #     print(k, output[k].shape)
        for i in range(len(self.L_keys)):
            k = self.L_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9 + len(self.F_keys)])).float()
        #     print(k, output[k].shape)
        # assert 0
        return output


class peyre_test_set(Dataset):

    def __init__(self, config, split='test'):

        self.data_dir = config.TRAIN.DATA_DIR
        self.split = split
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake_peyre.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_peyre.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys
        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        spatial, shape = load_pair_objects(info, sub_id, obj_id, rescale=False)
        feature, label = [], []
        with h5py.File(osp.join(self.data_dir, 'hake_peyre', str(im_id) + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key == 'FO':
                    feature.append(f[key][obj_id, ...])
                elif key == 'FR':
                    feature.append(f[key][cand_id, ...])
                elif key == 'FS':
                    feature.append(f[key][0, ...])
                else:
                    if key[-1] == 'l':
                        feature.append(f[key[:3]][sub_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature.append(f[key[:3]][sub_id, ...][1024:])
                    else:
                        feature.append(f[key][sub_id, ...])
        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label.append(np.zeros(P_num[key]))
            # if key in info:
            #     label[-1][info[key][cand_id]] = 1.

        obj_class = info['obj_classes'][obj_id] - 1
        key = int(info['filename'][-10:-4])
        hdet = info['obj_scores'][sub_id]
        odet = info['obj_scores'][obj_id]

        gt_obj = np.zeros(80)
        gt_obj[obj_class] = 1.

        begin, end = obj_range_padding[obj_class][0] - \
                     1, obj_range_padding[obj_class][1]
        rule = self.rule[begin:end, ...]

        gt_range = range(begin, end)

        return [key, spatial, shape, hdet, odet, obj_class, gt_obj, rule, gt_range] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}
        output['key'] = list(blob[0])
        output['spatial'] = torch.from_numpy(np.array(blob[1])).float()
        output['shape'] = torch.from_numpy(np.array(blob[2])).float()
        output['hdet'] = np.array(blob[3])
        output['odet'] = np.array(blob[4])
        output['obj_class'] = np.array(blob[5])
        output['gt_obj'] = torch.from_numpy(np.array(blob[6])).float()
        output['rule'] = torch.from_numpy(np.array(blob[7])).long()
        output['gt_range'] = torch.from_numpy(np.array(blob[8])).long()

        for i in range(len(self.F_keys)):
            k = self.F_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9])).float()
        #     print(k, output[k].shape)
        for i in range(len(self.L_keys)):
            k = self.L_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9 + len(self.F_keys)])).float()
        #     print(k, output[k].shape)
        # assert 0
        return output


class vcl_test_set(Dataset):

    def __init__(self, config, split='test'):

        self.data_dir = config.TRAIN.DATA_DIR
        self.split = split
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake_vcl_new.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_vcl_new.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys
        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        spatial, shape = load_pair_objects(info, sub_id, obj_id, rescale=False)
        feature, label = [], []
        with h5py.File(osp.join(self.data_dir, 'hake_vcl', str(im_id) + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key == 'FO':
                    feature.append(f[key][obj_id, ...])
                elif key == 'FR':
                    feature.append(f[key][cand_id, ...])
                elif key == 'FS':
                    feature.append(f[key][0, ...])
                else:
                    if key[-1] == 'l':
                        feature.append(f[key[:3]][sub_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature.append(f[key[:3]][sub_id, ...][1024:])
                    else:
                        feature.append(f[key][sub_id, ...])
                    # feature.append(f[key][sub_id, ...])
        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label.append(np.zeros(P_num[key]))
            if key in info:
                label[-1][info[key][cand_id]] = 1.

        obj_class = info['obj_classes'][obj_id] - 1
        key = int(info['filename'][-10:-4])
        hdet = info['obj_scores'][sub_id]
        odet = info['obj_scores'][obj_id]

        gt_obj = np.zeros(80)
        gt_obj[obj_class] = 1.

        begin, end = obj_range_padding[obj_class][0] - \
                     1, obj_range_padding[obj_class][1]
        rule = self.rule[begin:end, ...]

        gt_range = range(begin, end)

        return [key, spatial, shape, hdet, odet, obj_class, gt_obj, rule, gt_range] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}
        output['key'] = list(blob[0])
        output['spatial'] = torch.from_numpy(np.array(blob[1])).float()
        output['shape'] = torch.from_numpy(np.array(blob[2])).float()
        output['hdet'] = np.array(blob[3])
        output['odet'] = np.array(blob[4])
        output['obj_class'] = np.array(blob[5])
        output['gt_obj'] = torch.from_numpy(np.array(blob[6])).float()
        output['rule'] = torch.from_numpy(np.array(blob[7])).long()
        output['gt_range'] = torch.from_numpy(np.array(blob[8])).long()

        for i in range(len(self.F_keys)):
            k = self.F_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9])).float()
        #     print(k, output[k].shape)
        for i in range(len(self.L_keys)):
            k = self.L_keys[i]
            output[k] = torch.from_numpy(np.array(blob[i + 9 + len(self.F_keys)])).float()
        #     print(k, output[k].shape)
        # assert 0
        return output


class HICO_gt_test_set(Dataset):

    def __init__(self, config, split='test'):

        self.config = config
        self.data_dir = config.TEST.DATA_DIR
        self.split = split
        # # self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake.pkl'), 'rb'))
        # # self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '.pkl'), 'rb'))
        # if not config.TEST.get('EVAL_ALL') or not self.config.TEST.EVAL_ALL:
        #     self.db = pickle.load(
        #         open(osp.join(self.data_dir, 'db_' + self.split + '_hake_H_0.8_O_0.3_with_pvp_label.pkl'), 'rb'))
        #     self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_H_0.8_O_0.3.pkl'), 'rb'))
        # else:
        #     print('test all.')
        #     self.db = pickle.load(
        #         open(osp.join(self.data_dir, 'db_' + self.split + '_hake_with_pvp_label.pkl'), 'rb'))
        #     self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '.pkl'), 'rb'))
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_gt_' + self.split + '_hake_with_pvp_label.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_gt.pkl'), 'rb'))

        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys[:-1]

        rules = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # 600*n*6
        if isinstance(rules, dict):
            self.rule = rules['rules_index']
            self.rule_lens = rules['rules_len']
        else:
            self.rule = np.array(rules)

        # print(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):

        im_id, cand_id = self.cand[idx][:2]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]

        spatial, shape = load_pair_objects(info, sub_id, obj_id, rescale=False)
        feature, label = [], []

        for i in self.F_keys:
            feature.append([])

        with h5py.File(osp.join(self.data_dir, 'hake_gt', str(im_id) + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key == 'FH':
                    feature[i].append(f[key][sub_id, ...])
                elif key == 'FO':
                    feature[i].append(f[key][obj_id, ...])
                elif key == 'FS':
                    feature[i].append(f[key][0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][cand_id, ...])

        gt_parts = np.zeros(6)
        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label.append(info[key][cand_id])
            gt_parts[i] = np.max(label[-1][:-1])

        gt_parts = trans_6v_to_10v(gt_parts)

        begin, end = obj_range_padding[info['obj_classes'][obj_id] - 1][0] - 1, \
                     obj_range_padding[info['obj_classes'][obj_id] - 1][1]

        rules = self.rule[begin:end]
        rule_lens = None
        if hasattr(self, 'rule_lens'):
            rule_lens = self.rule_lens[begin:end]

        for i in range(len(self.F_keys)):
            feature[i] = np.array(feature[i])

        for i in range(len(self.L_keys)):
            label[i] = np.array(label[i])

        obj_class = info['obj_classes'][obj_id]
        key = int(info['filename'][-10:-4])
        hdet = info['obj_scores'][sub_id]
        odet = info['obj_scores'][obj_id]

        gt_obj = np.zeros(80)
        gt_obj[obj_class - 1] = 1.

        gt_range = range(begin, end)

        return [key, spatial, hdet, odet, obj_class - 1, gt_obj, gt_parts, rules, gt_range, rule_lens] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}
        output['key'] = np.array(blob[0])
        output['spatial'] = np.array(blob[1])
        output['hdet'] = np.array(blob[2])
        output['odet'] = np.array(blob[3])
        output['obj_class'] = np.array(blob[4])
        output['gt_obj'] = torch.from_numpy(np.array(blob[5])).float()
        output['gt_part'] = torch.from_numpy(np.array(blob[6])).float()
        output['rule'] = torch.from_numpy(np.array(blob[7])).long()
        output['gt_range'] = torch.from_numpy(np.array(blob[8])).long()

        if hasattr(self, 'rule_lens'):
            output['rule_lens'] = torch.from_numpy(np.array(blob[9])).long()

        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 10])).float()

        for i in range(len(self.L_keys)):
            output[self.L_keys[i]] = torch.from_numpy(np.array(blob[i + 10 + len(self.F_keys)])).float()

        return output


class AVA_im_train_set(Dataset):

    def __init__(self, config, split='train', train_mode=True):

        self.config = config
        self.data_dir = config.TRAIN.DATA_DIR
        self.split = split
        self.train_mode = train_mode
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake_ava.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_ava.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys

        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # num_verb * length *93
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):
        """ Intialize input matrices to empty """
        im_id, cand_id = self.cand[idx]
        info = self.db[im_id]
        spatial, shape = [], []
        feature, label = [], []
        for i in self.F_keys:
            feature.append([])
        for i in self.L_keys:
            label.append([])

        """ Fill the batch """

        layout, hw = info['boxes'][cand_id], [info['height'], info['width']]
        spatial.append(layout)
        shape.append(hw)
        with h5py.File(osp.join(self.data_dir, 'hake_ava', self.split, im_id + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key in ['FS', 'FO', 'FR']:
                    feature[i].append(f['FS'][0, 0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][0, cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][0, cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][0, cand_id, ...])

        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label[i].append(np.zeros(P_num[key]))
            if key in info and cand_id < len(info[key]) and -1 not in info[key][cand_id]:
                label[i][-1][np.array(info[key][cand_id]) - 1] = 1.

        for i in range(len(self.F_keys)):
            feature[i] = np.array(feature[i])
        for i in range(len(self.L_keys)):
            label[i] = np.array(label[i])
        spatial = np.array(spatial)
        shape = np.array(shape)

        rule = self.rule
        if self.config.MODEL.NUM_CLASS == 60:
            rule = rule[np.array(action) - 1, ...]

        return [spatial, shape, rule] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))
        output = {}
        output['spatial'] = torch.from_numpy(np.array(blob[0])).float()
        output['shape'] = torch.from_numpy(np.array(blob[1])).float()
        output['rule'] = torch.from_numpy(np.array(blob[2])).long()
        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 3])).float()
        for i in range(len(self.L_keys)):
            output[self.L_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 3 + len(self.F_keys)])).float()

        return output


class AVA_im_test_set(Dataset):

    def __init__(self, config, split='val', train_mode=True):

        self.config = config
        self.data_dir = config.TRAIN.DATA_DIR
        self.split = split
        self.train_mode = train_mode
        self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_hake_ava.pkl'), 'rb'))
        self.cand = pickle.load(open(osp.join(self.data_dir, 'cand_' + self.split + '_ava.pkl'), 'rb'))
        self.F_keys = config.MODEL.F_keys
        self.L_keys = config.MODEL.L_keys

        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # num_verb * length *93
        self.rule = np.array(self.rule)

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):
        """ Intialize input matrices to empty """
        im_id, cand_id = self.cand[idx]
        info = self.db[im_id]
        spatial, shape = [], []
        feature, label = [], []
        for i in self.F_keys:
            feature.append([])
        for i in self.L_keys:
            label.append([])

        """ Fill the batch """

        layout, hw = info['boxes'][cand_id], [info['height'], info['width']]
        spatial.append(layout)
        shape.append(hw)
        with h5py.File(osp.join(self.data_dir, 'hake_ava', self.split, im_id + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key in ['FS', 'FO', 'FR']:
                    feature[i].append(f['FS'][0, 0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][0, cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][0, cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][0, cand_id, ...])
                    # feature[i].append(f[key][0, cand_id, ...])

        for i in range(len(self.L_keys)):
            key = self.L_keys[i]
            label[i].append(np.zeros(P_num[key]))
            if key in info and cand_id < len(info[key]):
                label[i][-1][np.array(info[key][cand_id]) - 1] = 1.

        # for i in range(len(self.F_keys)):
        #     feature[i] = np.array(feature[i])
        # for i in range(len(self.L_keys)):
        #     label[i] = np.array(label[i])
        spatial = np.array(spatial)
        shape = np.array(shape)

        rule = self.rule
        if self.config.MODEL.NUM_CLASS == 60:
            rule = rule[np.array(action) - 1, ...]

        image_ids = im_id.split('/')[:2]
        image_ids[1] = image_ids[1].lstrip('0')
        info = [
                   image_ids[0],
                   image_ids[1].split('_')[0],
               ] + spatial[0].tolist()

        return [spatial, shape, rule, info] + feature + label

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))
        output = {}
        output['spatial'] = np.array(blob[0])
        output['shape'] = torch.from_numpy(np.array(blob[1])).float()
        output['rule'] = torch.from_numpy(np.array(blob[2])).long()
        output['info'] = list(blob[3])
        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 4])).float()
        for i in range(len(self.L_keys)):
            output[self.L_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 4 + len(self.F_keys)])).float()

        return output


class AVA_test_set_for_10v(Dataset):
    # transfer_im

    def __init__(self, config):
        self.config = config
        self.db = pickle.load(open(config.TEST.DATA_DIR, 'rb'))

        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # num_verb * length *93
        self.rule = np.array(self.rule)

        self.data_dir = 'Data'
        self.split = 'val'
        self.F_keys = config.MODEL.F_keys

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        info = self.db[idx][:-2]
        gt_pvp = self.db[idx][-2]
        cand_id = self.db[idx][-1]

        rule = self.rule
        if self.config.MODEL.NUM_CLASS == 60:
            rule = rule[np.array(action) - 1, ...]

        feature = []
        for i in self.F_keys:
            feature.append([])

        im_id = info[0] + '/' + info[1] + '_0.jpg'
        with h5py.File(osp.join(self.data_dir, 'hake_ava', self.split, im_id + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key in ['FS', 'FO', 'FR']:
                    feature[i].append(f['FS'][0, 0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][0, cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][0, cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][0, cand_id, ...])

        return [info, gt_pvp, rule] + feature

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        blob = list(zip(*data))

        output = {}

        output['info'] = list(blob[0])
        output['gt_part'] = torch.from_numpy(np.array(blob[1])).float()
        output['rule'] = torch.from_numpy(np.array(blob[2])).long()  # 600*76
        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 3])).float()

        return output


class AVA_train_set_for_10v(Dataset):
    # transfer_im

    def __init__(self, config):
        self.config = config
        self.db = pickle.load(open(config.TRAIN.DATA_DIR, 'rb'))
        self.rule = pickle.load(open(config.MODEL.RULE_DIR, 'rb'))  # num_verb * length *93
        self.rule = np.array(self.rule)

        self.data_dir = 'Data'
        self.split = 'train'
        self.F_keys = config.MODEL.F_keys

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        """ Intialize input matrices to empty """
        # self.rule: (600,10,k), k=6
        # batch['rule'] : (bz,18,10,k)

        label_index = self.db[idx][0]
        gt_label = np.zeros(self.config.MODEL.NUM_CLASS)
        gt_label[np.array(label_index).astype(np.int32) - 1] = 1

        gt_pvp = self.db[idx][1]

        feature = []
        for i in self.F_keys:
            feature.append([])
        im_id = self.db[idx][2]
        cand_id = self.db[idx][3]

        with h5py.File(osp.join(self.data_dir, 'hake_ava', self.split, im_id + '.h5'), 'r') as f:
            for i in range(len(self.F_keys)):
                key = self.F_keys[i]
                if key in ['FS', 'FO', 'FR']:
                    feature[i].append(f['FS'][0, 0, ...])
                else:
                    if key[-1] == 'l':
                        feature[i].append(f[key[:3]][0, cand_id, ...][:1024])
                    elif key[-1] == 'r':
                        feature[i].append(f[key[:3]][0, cand_id, ...][1024:])
                    else:
                        feature[i].append(f[key][0, cand_id, ...])

        rule = self.rule


        return [gt_label, gt_pvp, rule] + feature

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """

        blob = list(zip(*data))

        output = {}
        output['labels_a'] = torch.from_numpy(np.array(blob[0])).float()
        output['gt_part'] = torch.from_numpy(np.array(blob[1])).float()
        output['rule'] = torch.from_numpy(np.array(blob[2])).long()
        for i in range(len(self.F_keys)):
            output[self.F_keys[i]] = torch.from_numpy(np.concatenate(blob[i + 3])).float()

        return output
