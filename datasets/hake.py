import torch
import os.path as osp
import numpy as np
import json
import copy
from PIL import Image
from PIL import ImageFile

from util.vis_pasta import vis_image_pasta
from datasets import hake_meta

import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_hake_transformation(split):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    if split == 'train' or split == 'trainval':
        tfs = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            normalize
        ])
    elif split == 'val' or split == 'minival' or split == 'test':
        tfs = T.Compose([
            T.Resize((224, 224)),
            normalize
        ])
    else:
        raise KeyError("Unknown split")
    return tfs


class HAKEDataset:
    def __init__(self,
                 data_root,
                 ann_file,
                 tfs,
                 test_mode=False):
        self.data_root = data_root
        self.ann_file = ann_file
        self.tfs = tfs

        """
        gt_metas is used to generate ground truth;
        [(key, anno_key, num_classes, idx_for_binary_prediction),]
        idx < 0 means it is not used for part_binary_prediction
        """
        self.gt_metas = [
            ('verb', 'gt_verbs', hake_meta.num_classes['verb'], -1),
            ('foot', 'gt_pasta_foot', hake_meta.num_classes['foot'], 0),
            ('leg', 'gt_pasta_leg', hake_meta.num_classes['leg'], 1),
            ('hip', 'gt_pasta_hip', hake_meta.num_classes['hip'], 2),
            ('hand', 'gt_pasta_hand', hake_meta.num_classes['hand'], 3),
            ('arm', 'gt_pasta_arm', hake_meta.num_classes['arm'], 4),
            ('head', 'gt_pasta_head', hake_meta.num_classes['head'], 5),
        ]
        self.test_mode = test_mode

        self.load_annotations()

    def load_annotations(self):
        with open(self.ann_file, 'r') as f:
            hake_data = json.load(f)
        self.annotations = hake_data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        anno = copy.deepcopy(self.annotations[i])
        img_path = osp.join(self.data_root, anno['img_path'])

        gt_labels = {}
        img_metas = {'img_path': img_path}

        # handle a typo here
        img_metas['image_height'] = anno['image_heigh']
        img_metas['image_width'] = anno['image_width']

        gt_pasta_binary = np.zeros(hake_meta.num_classes['pasta_binary'], dtype=np.float32)

        for name, anno_key, num_classes, binary_i in self.gt_metas:
            gt_label_list = anno[anno_key]
            gt_label_np = np.zeros(num_classes, dtype=np.float32)
            gt_label_np[gt_label_list] = 1

            # set no_interaction to 0
            ignore_idx = hake_meta.ignore_idx[name]
            gt_label_np[ignore_idx] = 0
            gt_labels[name] = gt_label_np

            # assign binary label for part
            if binary_i >= 0:
                # the last one is ignore idx
                binary = (gt_label_np[:-1].sum() > 0).astype(np.float32)
                gt_pasta_binary[binary_i] = binary

        gt_labels['pasta_binary'] = gt_pasta_binary

        img = Image.open(img_path).convert('RGB')
        img = self.tfs(img)

        for key in gt_labels:
            gt_labels[key] = torch.from_numpy(gt_labels[key])

        return img, gt_labels, img_metas

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])

        labels = {}
        keys = batch[1][0].keys()
        for key in keys:
            labels[key] = torch.stack([x[key] for x in batch[1]])
        batch[1] = labels

        return tuple(batch)

    @staticmethod
    def calc_ap(gt, pred):
        hit = []
        idx = np.argsort(pred)[::-1]

        for i in idx:
            if gt[i] == 1:
                hit.append(1)
            else:
                hit.append(0)

        npos = gt.sum()
        assert npos > 0

        bottom = np.array(range(len(hit))) + 1
        hit    = np.cumsum(hit)
        rec    = hit / npos
        prec   = hit / bottom
        ap     = 0.0
        for i in range(11):
            mask = rec >= (i / 10.0)
            if np.sum(mask) > 0:
                ap += np.max(prec[mask]) / 11.0
        return ap

    def evaluate(self,
                 results):

        summary_res = {}

        def accum_pred(results, key):
            all_prob = []
            all_label = []
            for result in results:
                prob = result['prob'][key]  # [C]
                label = result['label'][key]
                all_prob.append(prob)
                all_label.append(label)
            all_prob = np.stack(all_prob)  # [N, C]
            all_label = np.stack(all_label)
            return all_prob, all_label

        eval_set = ['verb', 'foot', 'leg', 'head', 'hip', 'arm', 'hand', 'pasta_binary']
        for key in eval_set:
            names = hake_meta.class_names[key]
            ignore_idx = hake_meta.ignore_idx[key]
            all_probs, all_labels = accum_pred(results, key)
            assert all_probs.shape[1] == len(names)
            aps = [-1 for _ in range(len(names))]
            for i in range(len(names)):
                if i == ignore_idx:
                    continue
                else:
                    probs = all_probs[:, i]
                    labels = all_labels[:, i]
                    num_gt = sum(labels)
                    if num_gt < 1:
                        continue
                    ap = self.calc_ap(labels, probs)
                    aps[i] = ap
            out = {}
            for i in range(len(names)):
                if i != ignore_idx:
                    out[f'{names[i]}_AP'] = aps[i]
            aps = np.array(aps)
            mean_aps = aps[aps != -1].mean()
            out[f'* {key}_mAP'] = mean_aps
            summary_res[f'* {key}_mAP'] = mean_aps

        for k, v in summary_res.items():
            print(f'{k} : {v}')

        return summary_res

    def vis_test(self, img_path, result, output_dir):
        all_prob = []
        all_gt_prob = []
        all_names = []

        for key in result['prob'].keys():
            all_prob.append(result['prob'][key])
            all_gt_prob.append(result['label'][key])
            all_names.extend(copy.deepcopy(hake_meta.class_names[key]))

        all_prob = np.concatenate(all_prob)
        all_gt_prob = np.concatenate(all_gt_prob)
        vis_image_pasta(img_path, [(all_prob, all_names), (all_gt_prob, all_names)], out_dir=output_dir+'/vis', colors=['red', 'blue'])


def build(split, args):
    root = args.data_root
    PATHS = {
        "train": (root, root + '/hake_train_img.json'),
        "val": (root, root + '/hake_val_img.json'),
        'minival': (root, root + '/hake_minival_img.json'),
        'test': (root, root + '/hake_test_img.json')
    }

    img_folder, ann_file = PATHS[split]
    dataset = HAKEDataset(img_folder, ann_file, tfs=make_hake_transformation(split))
    return dataset
