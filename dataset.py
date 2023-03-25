import pickle
import numpy as np
import h5py
import os.path as osp
import torch
from torch.utils.data import Dataset
verb_trans = pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')

def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2

def jitter_boxes(box, width, height):
    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

    xt_min = -np.maximum((x2-x1)/5,1)
    xt_max = np.maximum((x2-x1)/5,1)
    yt_min = -np.maximum((y2-y1)/5,1)
    yt_max = np.maximum(1,(y2-y1)/5)
    x_trans = (xt_max-xt_min)*np.random.random() + xt_min
    y_trans = (yt_max-yt_min)*np.random.random() + yt_min

    x1_new = x1 + x_trans
    x2_new = x2 + x_trans
    y1_new = y1 + y_trans
    y2_new = y2 + y_trans 

    scale_factor = np.random.uniform(pow(2,-1/4), pow(2,1/4))
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

    x1_new,y1_new,x2_new,y2_new = clip_xyxy_to_image(x1_new, y1_new, x2_new, y2_new, height, width)

    if x2_new-x1_new < 1:
        x2_new = x2
        x1_new = x1

    if y2_new-y1_new < 1:
        y2_new = y2
        y1_new = y1

    jittered_box = np.array([x1_new,y1_new,x2_new,y2_new])

    return jittered_box

def flip_horizontal(box, width, height):
    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

    x1_flip = width-x2
    x2_flip = width-x1

    flip_box = np.array([x1_flip, y1, x2_flip, y2])

    return flip_box

def load_pair_objects(info, sub_id, obj_id, jittering=False):
    hbox   = info['boxes'][sub_id, :]
    obox   = info['boxes'][obj_id, :]
    width  = info['width']
    height = info['height']
    if jittering:
        hbox = jitter_boxes(hbox, width, height)
        obox = jitter_boxes(obox, width, height)

        if np.random.binomial(1,0.5):
            hbox = flip_horizontal(hbox, width, height)
            obox = flip_horizontal(obox, width, height)
    
    spatial = np.concatenate([hbox, obox])
    
    spatial[0] /= width
    spatial[1] /= height
    spatial[2] /= width
    spatial[3] /= height
    spatial[4] /= width
    spatial[5] /= height
    spatial[6] /= width
    spatial[7] /= height
    
    return spatial, np.array([width, height])


class HICO_test_set(Dataset):
    
    def __init__(self, data_dir, split='test'):
    
        self.data_dir     = data_dir
        self.split = split
        self.db    = pickle.load(open(osp.join(data_dir, 'db_' + self.split + '_feat.pkl'), 'rb'))
        self.cand  = pickle.load(open(osp.join(data_dir, 'candidates_' + self.split + '.pkl'), 'rb'))
        self.verb_trans   = verb_trans
        
        for key in self.db.keys():
            self.db[key]['H_mapping'], cnt = [], 0
            for i in range(self.db[key]['obj_classes'].shape[0]):
                if self.db[key]['obj_classes'][i] == 1:
                    self.db[key]['H_mapping'].append(cnt)
                    cnt += 1
                else:
                    self.db[key]['H_mapping'].append(-1)


    def __len__(self):
        return self.cand.shape[0]
    
    def __getitem__(self, idx):
        
        im_id, cand_id = self.cand[idx]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]
        
        spatial, shape = load_pair_objects(info, sub_id, obj_id)
        # There is a typo in the next line. If you extract test feature by yourself, please change str(int(self.db[im_id]['filename'][-10:-4])) to str(im_id). We will provide a more elegant fix in the late future.
        with h5py.File(osp.join(self.data_dir, 'feature', self.split, str(int(self.db[im_id]['filename'][-10:-4])) + '.h5'), 'r') as f:
            sub_vec = f['FH'][info['H_mapping'][sub_id], :]
            obj_vec = f['FO'][obj_id, :]
        with h5py.File(osp.join(self.data_dir, 'Union_feature', self.split, str(im_id) + '.h5'), 'r') as f:
            uni_vec = f['R'][info['feat_map'][cand_id], :]

        obj_class = info['obj_classes'][obj_id]
        key       = int(info['filename'][-10:-4])
        hdet      = info['obj_scores'][sub_id]
        odet      = info['obj_scores'][obj_id]
        
        if 'labels_sr' not in info:
            labels_s = np.zeros(117)
        else:
            labels_s  = info['labels_sr'][sub_id, :].toarray().astype(np.float32)
        if 'labels_r' not in info:
            labels_r  = np.zeros(117)
        else:
            labels_r  = info['labels_r'][cand_id, :].toarray().astype(np.float32)
        if 'labels_ro' not in info:
            labels_ro = np.zeros(600)
        else:
            labels_ro = info['labels_ro'][obj_id, :].toarray().astype(np.float32) # 600
        labels_sro = np.matmul(labels_s, self.verb_trans) * labels_ro

        return key, spatial, sub_vec, obj_vec, uni_vec, labels_s, labels_ro, labels_r, labels_sro, shape, hdet, odet, obj_class-1
    
    def collate_fn(self, data):
        key, spatial, sub_vec, obj_vec, uni_vec, labels_s, labels_ro, labels_r, labels_sro, shape, hdet, odet, obj_class = zip(*data)
        
        n = len(key)
        labels_r = np.concatenate(labels_r, axis=0)
        output = {}

        output['key']       = np.array(key)
        output['spatial']   = torch.from_numpy(np.array(spatial)).float()
        output['sub_vec']   = torch.from_numpy(np.array(sub_vec)).float()
        output['obj_vec']   = torch.from_numpy(np.array(obj_vec)).float()
        output['uni_vec']   = torch.from_numpy(np.array(uni_vec)).float()
        output['labels_s']  = torch.from_numpy(np.concatenate(labels_s, axis=0)).float()
        output['labels_ro'] = torch.from_numpy(np.concatenate(labels_ro, axis=0)).float()
        output['labels_r']  = torch.from_numpy(labels_r).float()
        output['labels_sro']= torch.from_numpy(np.concatenate(labels_sro, axis=0)).float()
        output['shape']     = torch.from_numpy(np.array(shape)).float()
        output['hdet']      = np.array(hdet)
        output['odet']      = np.array(odet)
        output['obj_class'] = np.array(obj_class)
        output['pos_ind']   = np.where(labels_r[:, 0] < 1)[0]

        return output


class HICO_train_set(Dataset):
    
    def __init__(self, config, split='trainval', train_mode=True):
        
        self.config       = config.TRAIN.DATASET
        self.data_dir     = config.TRAIN.DATA_DIR
        self.sampler_name = self.config.SAMPLER_NAME
        self.num_neg      = self.config.NUM_NEG
        self.split        = split
        self.train_mode   = train_mode
        self.jittering    = self.config.JITTERING
        self.ipt          = self.config.IPT
        self.db           = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '_with_pool.pkl'), 'rb'))
        self.cand_pos     = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + self.split + '.pkl'), 'rb'))
        self.cand_neg     = pickle.load(open(osp.join(self.data_dir, 'cand_negatives_' + self.split + '.pkl'), 'rb'))
        self.verb_trans   = verb_trans
        cand_cat = self.cand_neg[:,3]
        self.idx_match_object_candneg = {}
        for obj_cat in range(1, 81): 
            self.idx_match_object_candneg[obj_cat] = np.where(cand_cat==obj_cat)[0]

        cand_key = self.cand_neg[:,0]
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
        im_id, cand_id, sub, obj = self.cand_pos[idx]
        info = self.db[im_id]
        sub_id, obj_id = info['pair_ids'][cand_id]
        labels_s, labels_ro, labels_r, labels_sro, cand_info, spatial, sub_vec, obj_vec, uni_vec, shape = [], [], [], [], [], [], [], [], [], []

        cand_info.append(np.array([[im_id, cand_id, sub, obj]]))

        if self.sampler_name == 'priority_object':
            num_neg_sampled = 0

            if num_neg_sampled < self.num_neg and im_id in self.idx_match_image_candneg:
                idx_cand_match_image = self.idx_match_image_candneg[im_id]
                if len(idx_cand_match_image) > 0:
                    idx_pair_match_image     = list(self.cand_neg[idx_cand_match_image, 1])
                    obj_pair_match_image     = info['pair_ids'][idx_pair_match_image, 1]
                    idx_pair_match_image_obj = np.where(obj_pair_match_image == obj_id)[0]
                    idx_cand_match_image_obj = idx_cand_match_image[idx_pair_match_image_obj]
                    idx_neg = self.sample_random_negatives(idx_cand_match_image_obj, min(len(idx_cand_match_image_obj), self.num_neg-num_neg_sampled))
                    num_neg_sampled += len(idx_neg)
                    cand_info.append(self.cand_neg[idx_neg])

            if num_neg_sampled < self.num_neg:
                idx_match_object = self.idx_match_object_candneg[obj] 

                if len(idx_match_object)>0:
                    idx_neg = self.sample_random_negatives(idx_match_object, \
                                                           min(len(idx_match_object),self.num_neg-num_neg_sampled))
                    num_neg_sampled += len(idx_neg)
                    cand_info.append(self.cand_neg[idx_neg])

            if num_neg_sampled < self.num_neg:
                idx_neg = self.sample_random_negatives(np.arange(len(self.cand_neg)), \
                                                       self.num_neg-num_neg_sampled)
                num_neg_sampled += len(idx_neg)
                cand_info.append(self.cand_neg[idx_neg])

        elif self.sampler_name == 'random':
            num_neg_sampled = 0

            if num_neg_sampled < self.num_neg:
                idx_neg = np.random.choice(np.arange(len(self.cand_neg)), \
                                           self.num_neg-num_neg_sampled , replace=False)
                num_neg_sampled += len(idx_neg)
                cand_info.append(self.cand_neg[idx_neg])


        cand_info = np.concatenate(cand_info, axis=0)

        for j in range(cand_info.shape[0]):
            im_id, cand_id, sub, obj    = cand_info[j,:]
            info = self.db[im_id]
            sub_id, obj_id = info['pair_ids'][cand_id]
            labels_s.append(info['labels_sr'][sub_id, :].toarray().astype(np.float32))
            labels_ro.append(info['labels_ro'][obj_id, :].toarray().astype(np.float32))
            labels_r.append(info['labels_r'][cand_id, :].toarray().astype(np.float32))
            labels_sro.append(np.matmul(labels_s[-1], self.verb_trans) * labels_ro[-1])
            layout, hw = load_pair_objects(info, sub_id, obj_id, self.jittering)
            spatial.append(layout)
            shape.append(hw)
            if not self.ipt:
                with h5py.File(osp.join(self.data_dir, 'feature', self.split, str(im_id) + '.h5'), 'r') as f:
                    sub_vec.append(f['FH'][info['H_mapping'][sub_id], :])
                    obj_vec.append(f['FO'][obj_id, :])
            else:
                sub_ipt = info['pool'][sub_id][np.random.randint(0, info['pool'][sub_id].shape[0])]
                with h5py.File(osp.join(self.data_dir, 'feature', self.split, str(sub_ipt[0]) + '.h5'), 'r') as f:
                    sub_vec.append(f['FH'][self.db[sub_ipt[0]]['H_mapping'][sub_ipt[1]], :])
                obj_ipt = info['pool'][obj_id][np.random.randint(0, info['pool'][obj_id].shape[0])]
                with h5py.File(osp.join(self.data_dir, 'feature', self.split, str(obj_ipt[0]) + '.h5'), 'r') as f:
                    obj_vec.append(f['FO'][obj_ipt[1], :])
            l = im_id - im_id%400 + 1
            r = min(38119, l + 400)
            with h5py.File(osp.join(self.data_dir, 'Union_feature', self.split, str(l) + '_' + str(r) + '.h5'), 'r') as f:
                data = f['data'][...]
                mapping = pickle.load(open(self.data_dir+'/Union_feature/'+self.split+'/%d_%d.pkl' % (l, r), 'rb'))
                start = mapping[im_id]
                end = mapping[im_id+1] if im_id+1 in mapping else data.shape[0]
                uni_vec.append(data[start:end][cand_id, :])

        labels_s   = np.concatenate(labels_s, axis=0)
        labels_ro  = np.concatenate(labels_ro, axis=0)
        labels_r   = np.concatenate(labels_r, axis=0)
        labels_sro = np.concatenate(labels_sro, axis=0)
        spatial    = np.array(spatial)
        sub_vec    = np.array(sub_vec)
        obj_vec    = np.array(obj_vec)
        uni_vec    = np.array(uni_vec)
        shape      = np.array(shape)

        return cand_info, labels_s, labels_ro, labels_r, labels_sro, spatial, sub_vec, obj_vec, uni_vec, shape

    def collate_fn(self, data):
        cand_info, labels_s, labels_ro, labels_r, labels_sro, spatial, sub_vec, obj_vec, uni_vec, shape = zip(*data)
        output = {}
        
        labels_r = np.concatenate(labels_r, axis=0)
        output['cand_info']    = torch.from_numpy(np.concatenate(cand_info, axis=0))
        output['spatial']      = torch.from_numpy(np.concatenate(spatial, axis=0)).float()
        output['labels_s']     = torch.from_numpy(np.concatenate(labels_s, axis=0)).float()
        output['labels_r']     = torch.from_numpy(labels_r).float()
        output['labels_ro']    = torch.from_numpy(np.concatenate(labels_ro, axis=0)).float()
        output['labels_sro']   = torch.from_numpy(np.concatenate(labels_sro, axis=0)).float()
        output['sub_vec']      = torch.from_numpy(np.concatenate(sub_vec, axis=0)).float()
        output['obj_vec']      = torch.from_numpy(np.concatenate(obj_vec, axis=0)).float()
        output['uni_vec']      = torch.from_numpy(np.concatenate(uni_vec, axis=0)).float()
        output['shape']        = torch.from_numpy(np.concatenate(shape, axis=0)).float()
        n = output['shape'].shape[0]
        output['pos_ind']   = np.where(labels_r[:, 0] < 1)[0]

        return output
