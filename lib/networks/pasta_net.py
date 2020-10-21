#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Oct 22th, 2020            #
#############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import pickle
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ult.config  import cfg
sys.path.append(cfg.PASTA_LANGUAGE_MATRIX_PATH)

from ult.roi_pooling import ROIPooling2dPytorch as roi_pooling
from ult.ult_Generalized import write_hdf5
from .resnet_v1.resnetv1_torch import resnet50 as resnet50_v1
from .resnet_v1.resnetv1_torch import resnetv1
from ult.obj_80_768_averg_matrix import obj_matrix
from matrix_sentence import m_sentence
class pasta_res50(nn.Module):

    def __init__(self, args):
        super(pasta_res50, self).__init__()
        
        self.args          = args

        self.num_pasta0    = 16 # pasta0 ankle
        self.num_pasta1    = 15 # pasta1 knee
        self.num_pasta2    = 6 # pasta2 hip
        self.num_pasta3    = 34 # pasta3 hand
        self.num_pasta4    = 8 # pasta4 shoulder
        self.num_pasta5    = 14 # pasta5 head

        self.num_pastas    = [16, 15, 6, 34, 8, 14]
        self.split_pos = self.num_pastas
        self.split_pos = np.array(self.split_pos).cumsum(axis=0)[:-1]
        
        self.num_verbs     = 157
        self.num_vec       = 10
        self.num_fc        = 1024
        self.num_fc7_parts = [5120, 5120, 4096, 5120, 5120, 4096]

        self.pasta_mode    = self.args.pasta_mode if args is not None else 1
        self.pasta_trained = [int(pasta) for pasta in self.args.pasta_trained.split(',')] if args is not None else []

        self.pasta0_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_foot.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.pasta1_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_leg.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.pasta2_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_hip.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.pasta3_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_hand.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.pasta4_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_arm.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.pasta5_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_pasta_head.npy')).astype(np.float32).reshape(1, -1)).cuda()
        self.verb_weight = torch.from_numpy(np.load(os.path.join(cfg.PASTA_WEIGHTS_PATH,'weights_verb.npy')).astype(np.float32).reshape(1, -1)).cuda()

        self.pasta_matrix  = np.array(m_sentence, dtype='float32')[:,:1536]
        self.pasta_P0, self.pasta_P1, self.pasta_P2, self.pasta_P3, self.pasta_P4, self.pasta_P5 = np.split(self.pasta_matrix, self.split_pos, axis=0)
        self.pasta_P0 = torch.from_numpy(self.pasta_P0).cuda()
        self.pasta_P1 = torch.from_numpy(self.pasta_P1).cuda()
        self.pasta_P2 = torch.from_numpy(self.pasta_P2).cuda()
        self.pasta_P3 = torch.from_numpy(self.pasta_P3).cuda()
        self.pasta_P4 = torch.from_numpy(self.pasta_P4).cuda()
        self.pasta_P5 = torch.from_numpy(self.pasta_P5).cuda()

        self.resnet50 = resnet50_v1()
        self.resnet50.layer5 = copy.deepcopy(self.resnet50.layer4)

        # freeze layers
        def freeze(m):
            m.requires_grad=False
            
        for p in self.resnet50.parameters():
            p.requires_grad = False
        self.resnet50.conv1.padding = 0
        
        # tf: image_to_head
        self.image_to_head = nn.Sequential(
            # pad with 0
            nn.ConstantPad2d((0, 0, 3, 3), 0),
            nn.ConstantPad2d((3, 3, 0, 0), 0),

            # tmp_conv,
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,

            # pad with 0
            nn.ConstantPad2d((0, 0, 1, 1), 0),
            nn.ConstantPad2d((1, 1, 0, 0), 0),
            
            # max pooling
            torch.nn.MaxPool2d(kernel_size=[3, 3], stride=2),

            self.resnet50.layer1,
            self.resnet50.layer2,
            self.resnet50.layer3 )

        self.resnet_layer4   = self.resnet50.layer4
        self.crop_pool_layer = roi_pooling(output_size=(7,7), spatial_scale=float(1/16))

        self.fc7_parts   = nn.ModuleList(
                                            [
                                                nn.Sequential(
                                                    nn.Linear(self.num_fc7_parts[pasta_idx], 512),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(512,512),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.5)
                                                ) 
                                                for pasta_idx in range(6)
                                            ]
                                        )

        self.part_cls_scores = nn.ModuleList(
                                                [
                                                    nn.Linear(512, self.num_pastas[pasta_idx])
                                                    for pasta_idx in range(6)
                                                ]
                                            )

        for pasta_idx in range(6):
            train_bool = pasta_idx in self.pasta_trained
            for p in self.fc7_parts[pasta_idx].parameters():
                p.requires_grad = train_bool
            for p in self.part_cls_scores[pasta_idx].parameters():
                p.requires_grad = train_bool

        self.verb_cls_scores  = nn.Linear(3072, self.num_verbs)

        self.predictions = {}
        self.losses      = {}
        self.misc        = {}

        self.BCEcriterion_reduce = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
        self.BCEcriterion = torch.nn.BCEWithLogitsLoss().cuda()
        self.testMode = False

    '''
    @input:
        bottom: [1, 1024, H, W]
        rois  : [1, N, 5]
    @output: 
        crops : [N, 1024, 7, 7]
    '''

    def _crop_pool_layer(self, bottom, rois, max_pool=False):
        # implement it using stn
        # box to affine
        # input (x1,y1,x2,y2)
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        # print("rois.shape: ", rois.shape)
        rois = rois.detach()

        x1 = (rois[:, 1::4] / 16.0).squeeze(1)
        y1 = (rois[:, 2::4] / 16.0).squeeze(1)
        x2 = (rois[:, 3::4] / 16.0).squeeze(1)
        y2 = (rois[:, 4::4] / 16.0).squeeze(1)
        height = bottom.size(2)
        width  = bottom.size(3)

        # affine theta
        theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
        theta[:, 0, 0] = (x2 - x1) / (width - 1) 
        theta[:, 0 ,2] = (x1 + x2 - width + 1) / (width - 1) 
        theta[:, 1, 1] = (y2 - y1) / (height - 1) 
        theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1) 

        if max_pool:
            pre_pool_size = cfg.POOLING_SIZE * 2
            grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))

            all_roi = []
            for j in range(rois.size(0)):
                _grid = grid.narrow(0, j, 1)
                _roi_feature = F.grid_sample(bottom.view(1,bottom.size(1), bottom.size(2), bottom.size(3)), _grid)
                all_roi.append(_roi_feature)
            crops = torch.cat(all_roi)
            # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
            crops = F.max_pool2d(crops, 2, 2)
        else:
            grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))

            all_roi = []
            for j in range(rois.size(0)):
                _grid = grid.narrow(0, j, 1)
                _roi_feature = F.grid_sample(bottom.view(1,bottom.size(1), bottom.size(2), bottom.size(3)), _grid)
                all_roi.append(_roi_feature)
            crops = torch.cat(all_roi)
            # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)

        return crops

    def res5(self, pool5_H):
        fc7_H = self.resnet_layer4(pool5_H)
        fc7_H = torch.mean(fc7_H, [2, 3])
        return fc7_H
    
    # extract part ROIs
    def ROI_for_parts(self, head, fc5_H, fc5_S, P_boxes):
        pool5_P0 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 0, :]),[2,3])
        pool5_P1 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 1, :]),[2,3])
        pool5_P2 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 2, :]),[2,3])
        pool5_P3 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 3, :]),[2,3])
        pool5_P4 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 4, :]),[2,3])
        pool5_P5 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 5, :]),[2,3])
        pool5_P6 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 6, :]),[2,3])
        pool5_P7 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 7, :]),[2,3])
        pool5_P8 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 8, :]),[2,3])
        pool5_P9 = torch.mean(self._crop_pool_layer(head, P_boxes[:, 9, :]),[2,3])

        fc5_S  = fc5_S.repeat([pool5_P0.shape[0], 1])
        fc5_P0 = torch.cat([pool5_P0, pool5_P3, fc5_H, fc5_S], 1)
        fc5_P1 = torch.cat([pool5_P1, pool5_P2, fc5_H, fc5_S], 1)
        fc5_P2 = torch.cat([pool5_P4, fc5_H, fc5_S], 1)
        fc5_P3 = torch.cat([pool5_P6, pool5_P9, fc5_H, fc5_S], 1)
        fc5_P4 = torch.cat([pool5_P7, pool5_P8, fc5_H, fc5_S], 1)
        fc5_P5 = torch.cat([pool5_P5, fc5_H, fc5_S], 1)

        return fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5

    # classification for each pasta
    def part_classification(self, pool5_P, part_idx):

        fc7_P       = self.fc7_parts[part_idx](pool5_P)
        cls_score_P = self.part_cls_scores[part_idx](fc7_P)
        cls_prob_P  = torch.sigmoid(cls_score_P)   

        return cls_score_P, cls_prob_P, fc7_P

    # global pasta classification
    def pasta_classification(self, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5):
        # ipdb.set_trace()
        cls_score_P0, cls_prob_P0, fc7_P0 = self.part_classification(pool5_P0, 0)
        cls_score_P1, cls_prob_P1, fc7_P1 = self.part_classification(pool5_P1, 1)
        cls_score_P2, cls_prob_P2, fc7_P2 = self.part_classification(pool5_P2, 2)
        cls_score_P3, cls_prob_P3, fc7_P3 = self.part_classification(pool5_P3, 3)
        cls_score_P4, cls_prob_P4, fc7_P4 = self.part_classification(pool5_P4, 4)
        cls_score_P5, cls_prob_P5, fc7_P5 = self.part_classification(pool5_P5, 5)
        
        self.predictions["cls_score_pasta0"]  = cls_score_P0
        self.predictions["cls_score_pasta1"]  = cls_score_P1
        self.predictions["cls_score_pasta2"]  = cls_score_P2
        self.predictions["cls_score_pasta3"]  = cls_score_P3
        self.predictions["cls_score_pasta4"]  = cls_score_P4
        self.predictions["cls_score_pasta5"]  = cls_score_P5
        cls_prob_PaSta = torch.cat([cls_prob_P0, cls_prob_P1, cls_prob_P2, cls_prob_P3, cls_prob_P4, cls_prob_P5], 1)
        self.predictions["cls_prob_PaSta"]    = cls_prob_PaSta
        self.predictions['cls_score_PaSta']   = torch.cat([cls_score_P0, cls_score_P1, cls_score_P2, cls_score_P3, cls_score_P4, cls_score_P5], 1)
            
        return fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5

    # extract language feature
    def language_head(self):

        fc7_L = torch.cat([
            torch.matmul(self.predictions["cls_prob_PaSta"][:, :self.split_pos[0]], self.pasta_P0),
            torch.matmul(self.predictions["cls_prob_PaSta"][:, self.split_pos[0]:self.split_pos[1]], self.pasta_P1),
            torch.matmul(self.predictions["cls_prob_PaSta"][:, self.split_pos[1]:self.split_pos[2]], self.pasta_P2),
            torch.matmul(self.predictions["cls_prob_PaSta"][:, self.split_pos[2]:self.split_pos[3]], self.pasta_P3),
            torch.matmul(self.predictions["cls_prob_PaSta"][:, self.split_pos[3]:self.split_pos[4]], self.pasta_P4),
            torch.matmul(self.predictions["cls_prob_PaSta"][:, self.split_pos[4]:], self.pasta_P5),
        ], 1)

        return fc7_L

    def verb_classification(self, fc7_P):

        cls_score_verb = self.verb_cls_scores(fc7_P)
        cls_prob_verb = torch.sigmoid(cls_score_verb)
        
        self.predictions["cls_score_verb"]   = cls_score_verb
        self.predictions["cls_prob_verb"]   = cls_prob_verb

        return     

    def forward(self, blobs, debug=False, mode='default'):
        image   = blobs['image'].float().cuda()
        H_boxes = blobs['H_boxes'].float().cuda()
        P_boxes = blobs['P_boxes'].float().cuda()
        if not mode == 'inference':
            self.misc['dataset'] = blobs['dataset']
        self.H_num = int(blobs['H_num'])
        
        if self.training:
            self.gt_class_P0 = blobs['gt_class_P0'].float().cuda() # target ankle status
            self.gt_class_P1 = blobs['gt_class_P1'].float().cuda() # target knee status
            self.gt_class_P2 = blobs['gt_class_P2'].float().cuda() # target hip status
            self.gt_class_P3 = blobs['gt_class_P3'].float().cuda() # target hand status
            self.gt_class_P4 = blobs['gt_class_P4'].float().cuda() # target shoulder status
            self.gt_class_P5 = blobs['gt_class_P5'].float().cuda() # target head status
            self.gt_verb     = blobs['gt_verb'].float().cuda() # target verb
            H_boxes = H_boxes.squeeze(0)
            P_boxes = P_boxes.squeeze(0)
        head = self.image_to_head(image)
        pool5_H = self._crop_pool_layer(head, H_boxes)
        fc5_H = self.res5(pool5_H)
        fc5_S = torch.mean(head, [2, 3])
        fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5 = self.ROI_for_parts(head, fc5_H, fc5_S, P_boxes)

        fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5 = self.pasta_classification(fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5)
        fc7_P = torch.cat([fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5], 1)
        self.verb_classification(fc7_P)

        fc7_L = self.language_head()
        PaSta_feat = torch.cat([fc7_P, fc7_L], 1)
        
        if mode == 'inference':
            return PaSta_feat, self.predictions["cls_prob_PaSta"], self.predictions["cls_prob_verb"]
        return True

    def add_loss(self, debug=False):
        # here use cls_score, not cls_prob
        cls_score_P0 = self.predictions["cls_score_pasta0"]
        cls_score_P1 = self.predictions["cls_score_pasta1"]
        cls_score_P2 = self.predictions["cls_score_pasta2"]
        cls_score_P3 = self.predictions["cls_score_pasta3"]
        cls_score_P4 = self.predictions["cls_score_pasta4"]
        cls_score_P5 = self.predictions["cls_score_pasta5"]
        
        label_P0     = self.gt_class_P0.squeeze(0)
        label_P1     = self.gt_class_P1.squeeze(0)
        label_P2     = self.gt_class_P2.squeeze(0)
        label_P3     = self.gt_class_P3.squeeze(0)
        label_P4     = self.gt_class_P4.squeeze(0)
        label_P5     = self.gt_class_P5.squeeze(0)

        P0_cross_entropy = self.BCEcriterion_reduce(cls_score_P0[:self.H_num, :], label_P0[:self.H_num, :])
        P1_cross_entropy = self.BCEcriterion_reduce(cls_score_P1[:self.H_num, :], label_P1[:self.H_num, :])
        P2_cross_entropy = self.BCEcriterion_reduce(cls_score_P2[:self.H_num, :], label_P2[:self.H_num, :])
        P3_cross_entropy = self.BCEcriterion_reduce(cls_score_P3[:self.H_num, :], label_P3[:self.H_num, :])
        P4_cross_entropy = self.BCEcriterion_reduce(cls_score_P4[:self.H_num, :], label_P4[:self.H_num, :])
        P5_cross_entropy = self.BCEcriterion_reduce(cls_score_P5[:self.H_num, :], label_P5[:self.H_num, :])
        
        P0_cross_entropy = torch.mean(P0_cross_entropy.mul(self.pasta0_weight))
        P1_cross_entropy = torch.mean(P1_cross_entropy.mul(self.pasta1_weight))
        P2_cross_entropy = torch.mean(P2_cross_entropy.mul(self.pasta2_weight))
        P3_cross_entropy = torch.mean(P3_cross_entropy.mul(self.pasta3_weight))
        P4_cross_entropy = torch.mean(P4_cross_entropy.mul(self.pasta4_weight))
        P5_cross_entropy = torch.mean(P5_cross_entropy.mul(self.pasta5_weight))
        PaSta_cross_entropy = P0_cross_entropy + P1_cross_entropy + P2_cross_entropy + P3_cross_entropy + P4_cross_entropy + P5_cross_entropy
        
        # to do: pos_weight
        self.losses['pasta0_cross_entropy']  = P0_cross_entropy
        self.losses['pasta1_cross_entropy']  = P1_cross_entropy
        self.losses['pasta2_cross_entropy']  = P2_cross_entropy
        self.losses['pasta3_cross_entropy']  = P3_cross_entropy
        self.losses['pasta4_cross_entropy']  = P4_cross_entropy
        self.losses['pasta5_cross_entropy']  = P5_cross_entropy

        
        label_verb   = self.gt_verb.squeeze(0)
        cls_score_verb = self.predictions['cls_score_verb']
        verb_cross_entropy = self.BCEcriterion_reduce(cls_score_verb, label_verb)
        verb_cross_entropy = torch.mean(verb_cross_entropy.mul(self.verb_weight))
        self.losses['verb_cross_entropy']  = verb_cross_entropy
        
        if self.pasta_mode == 1: # all
            loss = PaSta_cross_entropy + verb_cross_entropy
        elif self.pasta_mode == 2: # pasta only
            loss = 0
            for pasta in self.pasta_trained:
                loss += self.losses['pasta%d_cross_entropy' % pasta]
        elif self.pasta_mode == 3: # verb only
            loss = verb_cross_entropy
        else:
            raise NotImplementedError
        self.losses['total_loss'] = loss

        return loss

