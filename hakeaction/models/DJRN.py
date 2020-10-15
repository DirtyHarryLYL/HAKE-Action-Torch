import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable
import numpy as np
import copy
import pickle
import h5py
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs.config_DJRN import cfg
from core.utils.roi_pooling import ROIPooling2dPytorch as roi_pooling
from core.backbones.resnet_v1.resnetv1_torch import resnet50 as resnet50_v1
from datasets.utils.HICO_DET_utils import HO_weight, binary_weight

'''
@input:
    name:           shape:
    image           [1, None, None, 3]
    spatial         [None, 64, 64, 3]
    H_boxes         [None, 5]
    O_boxes         [None, 5]
    gt_class_HO     [None, 600]
'''
class DJRN_ResNet50(nn.Module):
    def __init__(self):
        super(DJRN_ResNet50, self).__init__()
        self.cfg = cfg

        '''
            Global settings
        '''
        self.num_classes = 600  # num of HOI classes
        self.num_fc      = 1024
        self.num_binary  = 2

        '''
            Backbone settings
        '''
        # self.resnet50 = torch.hub.load('pytorch/vision:v0.4.0', 'resnet50', pretrained=True)
        self.resnet50 = resnet50_v1()
        self.resnet50.conv1.padding = 0
        self.resnet50.layer5 = copy.deepcopy(self.resnet50.layer4)

        '''
            Network Architecture Definition
        '''
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
        self.resnet_layer5   = copy.deepcopy(self.resnet50.layer4)
        self.crop_pool_layer = roi_pooling(output_size=(7,7), spatial_scale=float(1/16))

        '''
            spatial configuration, conv-pool-conv-pool-flatten
            input:  [num_pos_neg, 2, 64, 64]
            output: [num_pos_neg, 5408]
        '''
        self.sp_to_head = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0),

            nn.Conv2d(64, 32, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0), )

        # whole image feature
        self.head_phi = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU() )
        
        self.head_g = nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1)),
            nn.ReLU() )

        self.attention_pool_layer_H_network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.8))

        self.attention_pool_layer_O_network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.8))

        self.attention_pool_layer_sp_network = nn.Sequential(
            nn.Linear(12288, 32),
            nn.ReLU(),
            nn.Dropout(0.8))

        self.head_bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU())

        self.head_bottleneck_sp = nn.Sequential(
            nn.Conv2d(32, 256, (1, 1)),
            nn.ReLU())

        self.fc8_SH = nn.Sequential(
            nn.Linear(3072, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc9_SH = nn.Sequential(
            nn.Linear(1024, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc8_SO = nn.Sequential(
            nn.Linear(3072, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc9_SO = nn.Sequential(
            nn.Linear(1024, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.Concat_SHsp = nn.Sequential(
            nn.Linear(2560, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc7_SHsp = nn.Sequential(
            nn.Linear(self.num_fc, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.pool2_flat_pose_map = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 16, (5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),)

        self.fc8_binary_1 = nn.Sequential(
            nn.Linear(11184, self.num_fc),
            nn.ReLU(),
            nn.Dropout(cfg.TRAIN_DROP_OUT_BINARY))

        self.fc8_binary_2 = nn.Sequential(
            nn.Linear(3072, self.num_fc),
            nn.ReLU(),
            nn.Dropout(cfg.TRAIN_DROP_OUT_BINARY))

        self.fc9_binary = nn.Sequential(
            nn.Linear(2048, self.num_fc),
            nn.ReLU(),
            nn.Dropout(cfg.TRAIN_DROP_OUT_BINARY))

        self.body_to_head = nn.Sequential(
            nn.Linear(85, self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(self.num_fc, self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        self.fc3_G = nn.Sequential(
            nn.Linear(1280, self.num_fc),
            nn.ReLU(),
            nn.Dropout(0.5) )
        
        self.A_3D = nn.Linear(self.num_fc, 17)
        
        self.fc_sp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5) )
        
        '''
            classification layers
        '''
        self.cls_score_H      = nn.Linear(1024, self.num_classes)
        self.cls_score_O      = nn.Linear(1024, self.num_classes)
        self.cls_score_sp     = nn.Linear(1024, self.num_classes)
        self.cls_score_D      = nn.Linear(1536, self.num_classes)
        self.cls_score_J      = nn.Linear(4608, self.num_classes)
        self.cls_score_BSH    = nn.Linear(2048, self.num_classes)

        self.predictions   = {}
        self.losses        = {}
        self.HO_weight     = torch.tensor(HO_weight).cuda()
        self.binary_weight = torch.tensor(binary_weight).cuda()

        '''
            criterion settings
        '''
        self.BCEcriterion = torch.nn.BCEWithLogitsLoss().cuda()
        self.testMode     = False
        
    '''
        Freeze layers
    '''
    def freezeParts(self, partsToFreeze):
        def freeze(m):
            m.requires_grad = False
            
        for part in partsToFreeze:
            assert hasattr(self, part), "DJRN does not have attribute:{}".format(part)
            freezeCmd = "self.{}.apply(freeze)".format(part)
            print("cmd to exex: ", freezeCmd)
            exec(freezeCmd)

        return
    
    def _crop_pool_layer(self, bottom, rois, max_pool=False):
        """
            @input:
                bottom: [1, 1024, H, W]
                rois  : [1, N, 5] (x1,y1,x2,y2)
            @output: 
                crops : [N, 1024, 7, 7]
                
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1       ]
        """

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
            crops = F.max_pool2d(crops, 2, 2)
        else:
            grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))

            all_roi = []
            for j in range(rois.size(0)):
                _grid = grid.narrow(0, j, 1)
                _roi_feature = F.grid_sample(bottom.view(1,bottom.size(1), bottom.size(2), bottom.size(3)), _grid)
                all_roi.append(_roi_feature)
            crops = torch.cat(all_roi)
            
        return crops

    def res5(self, pool5_H, pool5_O):
        fc7_H = self.resnet_layer4(pool5_H)
        fc7_H = torch.mean(fc7_H, [2, 3])

        fc7_O = self.resnet_layer5(pool5_O)
        fc7_O = torch.mean(fc7_O, [2, 3])

        return fc7_H, fc7_O

    def attention_pool_layer_H(self, bottom, fc7_H, debug=False):
        fc_1 = self.attention_pool_layer_H_network(fc7_H)
        fc_1 = fc_1.view([fc_1.size()[0], fc_1.size()[1], 1, 1])
        att = torch.mean(bottom.mul(fc_1), 1)
        att = att.unsqueeze(1)

        if debug:
            print("==> [attention_pool_layer_H] bottom.shape: ", bottom.shape)
            print("==> [attention_pool_layer_H] fc_1.shape: ", fc_1.shape)
            print("==> [attention_pool_layer_H] att.shape: ", att.shape)

        return att

    def attention_pool_layer_O(self, bottom, fc7_O, debug=False):
        fc_1 = self.attention_pool_layer_O_network(fc7_O)
        fc_1 = fc_1.view([fc_1.size()[0], fc_1.size()[1], 1, 1])
        att = torch.mean(bottom.mul(fc_1), 1)
        att = att.unsqueeze(1)

        if debug:
            print("==> [attention_pool_layer_O] bottom.shape: ", bottom.shape)
            print("==> [attention_pool_layer_O] fc_1.shape: ", fc_1.shape)
            print("==> [attention_pool_layer_O] att.shape: ", att.shape)

        return att

    def attention_pool_layer_sp(self, bottom, fc7_H, fc7_O, debug=False):
        # print(bottom.shape) # 16, 32, 13, 13
        key = torch.flatten(bottom, start_dim=1)
        key = torch.cat([key, fc7_H, fc7_O], axis=1) # 16 x 9504

        fc_1 = self.attention_pool_layer_sp_network(key) 
        fc_1 = fc_1.view([fc_1.size()[0], fc_1.size()[1], 1, 1])

        att = torch.mean(bottom.mul(fc_1), 1)
        att = att.unsqueeze(1)

        if debug:
            print("==> [attention_pool_layer_sp] bottom.shape: ", bottom.shape)
            print("==> [attention_pool_layer_sp] fc_1.shape: ", fc_1.shape)
            print("==> [attention_pool_layer_sp] att.shape: ", att.shape)

        return att

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, debug=False):

        fc7_SH = torch.mean(pool5_SH, axis=[2, 3])
        fc7_SO = torch.mean(pool5_SO, axis=[2, 3])
        Concat_SH = torch.cat([fc7_H, fc7_SH], 1) # -> [N, 3072]
        
        if debug:
            print("==> [head_to_tail] pool5_SH.shape: ", pool5_SH.shape)
            print("==> [head_to_tail] fc7_SH.shape: ", fc7_SH.shape)
            print("==> [head_to_tail] fc7_SO.shape: ", fc7_SO.shape)
            print("==> [head_to_tail] fc7_H.shape: ", fc7_H.shape)
            print("==> [head_to_tail] Concat_SH.shape: ", Concat_SH.shape)

        fc8_SH = self.fc8_SH(Concat_SH)
        fc9_SH = self.fc9_SH(fc8_SH)

        Concat_SO = torch.cat([fc7_O, fc7_SO], 1)

        if debug:
            print("==> [head_to_tail] fc7_O.shape: ", fc7_O.shape)
            print("==> [head_to_tail] fc7_SO.shape: ", fc7_SO.shape)
            print("==> [head_to_tail] Concat_SO.shape: ", Concat_SO.shape)

        fc8_SO = self.fc8_SO(Concat_SO)
        fc9_SO = self.fc9_SO(fc8_SO)
        Concat_SHsp   = torch.cat([fc7_H, sp], 1)

        if debug:
            print("==> [head_to_tail] sp.shape: ", sp.shape)
            print("==> [head_to_tail] Concat_SHsp.shape: ", Concat_SHsp.shape)

        Concat_SHsp = self.Concat_SHsp(Concat_SHsp)
        fc7_SHsp    = self.fc7_SHsp(Concat_SHsp)

        return fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, debug=False):
        cls_score_H  = self.cls_score_H(fc7_H)
        cls_prob_H   = torch.sigmoid(cls_score_H)   
        # torch.reshape(cls_prob_H, (1, self.num_classes))
 
        cls_score_O  = self.cls_score_O(fc7_O)
        cls_prob_O   = torch.sigmoid(cls_score_O)   
        # torch.reshape(cls_prob_O, (1, self.num_classes))

        cls_score_sp = self.cls_score_sp(fc7_SHsp)
        cls_prob_sp  = torch.sigmoid(cls_score_sp)
        
        if debug:
            print("==> [region_classification] cls_prob_H.shape: ", cls_prob_H.shape)
            print("==> [region_classification] cls_prob_O.shape: ", cls_prob_O.shape)
            print("==> [region_classification] cls_prob_sp.shape: ", cls_prob_sp.shape)
        # torch.reshape(cls_prob_sp, (1, self.num_classes))

        self.predictions["cls_score_H"]  = cls_score_H
        self.predictions["cls_prob_H"]   = cls_prob_H
        self.predictions["cls_score_O"]  = cls_score_O
        self.predictions["cls_prob_O"]   = cls_prob_O
        self.predictions["cls_score_sp"] = cls_score_sp
        self.predictions["cls_prob_sp"]  = cls_prob_sp

        # late fusion of predictions 
        # self.predictions["cls_prob_R"]    = cls_prob_sp * (cls_prob_H + cls_prob_O)
        self.predictions["cls_prob_R"]    = cls_prob_sp

        return cls_prob_H, cls_prob_O, cls_prob_sp

    def attention_3D(self, fc2_G, pc_att_map):
        fc3_G     = self.fc3_G(fc2_G)
        A_3D      = self.A_3D(fc3_G)
        A_3D      = A_3D.softmax(-1)
        att_3D    = A_3D.matmul(pc_att_map[0])
        att_shape = att_3D.shape
        att_3D    = att_3D.view([att_shape[0], att_shape[1], -1]) 
        
        self.predictions['A_3D'] = A_3D
        return att_3D
        
    def attention_2D(self, att_2D, att_2D_map):
        att_tmp = att_2D_map.mul(att_2D)
        A_2D    = torch.mean(att_tmp, axis=[2, 3])
        bottom  = torch.mean(A_2D)
        A_2D    = A_2D / bottom
        
        self.predictions['A_2D'] = A_2D
        return
    
    def space_classification(self, fc3_G):
        cls_score_D = self.cls_score_D(fc3_G)
        cls_prob_D  = torch.sigmoid(cls_score_D) 

        self.predictions["cls_score_D"]  = cls_score_D
        self.predictions["cls_prob_D"]   = cls_prob_D
        return
        
    def joint_classification(self, fc_J):
        cls_score_J  = self.cls_score_J(fc_J)
        cls_prob_J   = torch.sigmoid(cls_score_J) 

        self.predictions["cls_score_J"]  = cls_score_J
        self.predictions["cls_prob_J"]   = cls_prob_J
        return
         
    def triplet_align(self, fc2_C, sp):
        fc_sp     = self.fc_sp(sp)
        label_HO  = self.gt_class_HO.squeeze(0)
        label_HO_ = torch.transpose(label_HO, 1, 0)
        sim       = label_HO.matmul(label_HO_)
        zeros     = torch.zeros_like(sim, dtype=sim.dtype)
        ones      = torch.ones_like(sim,  dtype=sim.dtype)
        pos_mask  = torch.where(sim > 0, ones, zeros)
        neg_mask  = torch.where(sim < 1, ones, zeros)
        
        fc2_C_    = torch.transpose(fc2_C, 1, 0)
        dot       = fc_sp.matmul(fc2_C_)
        dot       = torch.sigmoid(dot)
        
        self.losses['L_tri'] = torch.mean(dot * neg_mask + (1 - dot) * pos_mask)
        return
        
    def forward(self, blobs, debug=False):
        image      = blobs['image']
        H_boxes    = blobs['H_boxes']
        O_boxes    = blobs['O_boxes']
        spatial    = blobs['sp']
        att_2D_map = blobs['att_2D_map'] # [None, 64, 64, 17]
        pc         = blobs['pc'][0]
        pc_att_map = blobs['pc_att_map']
        smplx      = blobs['smplx']
        self.H_num = int(blobs['H_num']) # e.g., 16

        # for k,v in blobs.items():
        #     try:
        #         print(k, v.shape)
        #     except:
        #         continue

        if not self.testMode:
            # train mode, get gt in blobs
            self.gt_class_HO = blobs['gt_class_HO']

        head = self.image_to_head(image)

        try:  # try crop and pool
            if self.testMode: 
                pool5_H = self._crop_pool_layer(head, H_boxes)
                pool5_O = self._crop_pool_layer(head, O_boxes)
            else:
                pool5_H = self._crop_pool_layer(head, H_boxes.squeeze(0))
                pool5_O = self._crop_pool_layer(head, O_boxes.squeeze(0))
        except:
            # crop fail
            return False

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O)

        # whole image feature
        head_phi = self.head_phi(head)
        head_g   = self.head_g(head)

        if self.testMode:
            pool2_sp = self.sp_to_head(spatial.permute(0,3,1,2))
        else:
            pool2_sp = self.sp_to_head(spatial[0].permute(0,3,1,2)) # [1, 16, 64, 64, 3] -> [16, 3, 64, 64]

        att_2D_map   = att_2D_map.squeeze(0).permute(0,3,1,2)
        att_2D_map   = nn.AvgPool2d((2, 2), padding=0)(att_2D_map) * 4
        att_2D_map   = nn.AvgPool2d((2, 2), padding=0)(att_2D_map) * 4

        Att_H        = self.attention_pool_layer_H(head_phi, fc7_H)
        Att_H_shape  = Att_H.shape
        Att_H        = Att_H.view(Att_H_shape[0], Att_H_shape[1], -1)
        Att_H        = Att_H.softmax(-1) # norm H
        Att_H        = Att_H.view(Att_H_shape)
        att_head_H   = head_g.mul(Att_H)

        Att_O        = self.attention_pool_layer_O(head_phi, fc7_O)
        Att_O_shape  = Att_O.shape
        Att_O        = Att_O.view(Att_O_shape[0], Att_O_shape[1], -1)
        Att_O        = Att_O.softmax(-1) # norm O
        Att_O        = Att_O.view(Att_O_shape)
        att_head_O   = head_g.mul(Att_O)

        Att_sp       = self.attention_pool_layer_sp(pool2_sp, fc7_H, fc7_O)
        Att_sp_shape = Att_sp.shape
        Att_sp       = Att_sp.view(Att_sp_shape[0], Att_sp_shape[1], -1)
        Att_sp       = Att_sp.softmax(-1) # norm sp
        Att_sp       = Att_sp.view(Att_sp_shape)
        att_head_sp  = pool2_sp.mul(Att_sp)
        self.attention_2D(Att_sp, att_2D_map)

        pool5_SH     = self.head_bottleneck(att_head_H)
        pool5_SO     = self.head_bottleneck(att_head_O)
        pool5_Ssp    = self.head_bottleneck_sp(att_head_sp)
        pool5_sp     = self.head_bottleneck_sp(pool2_sp)
        pool5_SP     = torch.cat([pool5_Ssp, pool5_sp], axis=1)
        sp           = torch.mean(pool5_SP, axis=[2, 3])

        fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO = self.head_to_tail(
            fc7_H, fc7_O, pool5_SH, pool5_SO, sp)

        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(
            fc9_SH, fc9_SO, fc7_SHsp)

        # 3D branch
        fc2_B  = self.body_to_head(smplx[0])
        fc1_C  = torch.mean(pc, axis=[1])
        fc2_G  = torch.cat([fc2_B, fc1_C], axis=1)
        
        att_3D = self.attention_3D(fc2_G, pc_att_map)
        fc1_SC = torch.mean(pc.mul(att_3D), axis=1)
        fc2_C  = torch.cat([fc1_SC, fc1_C], axis=1)
        
        fc3_G  = torch.cat([fc2_G, fc1_SC], axis=1)
        fc_J   = torch.cat([fc3_G, fc9_SH, fc9_SO, fc7_SHsp], axis=1)
        
        self.space_classification(fc3_G)
        self.joint_classification(fc_J)
        self.predictions['cls_prob_HO'] = self.predictions['cls_prob_D'] + self.predictions['cls_prob_J'] + self.predictions['cls_prob_R']      

        # in test mode, get inference results
        if self.testMode: 
            return self.predictions["cls_prob_HO"], cls_prob_H, cls_prob_O, cls_prob_sp
            # return self.predictions["cls_prob_R"], cls_prob_H, cls_prob_O, cls_prob_sp

        self.triplet_align(fc2_C, sp)

        # concat fc2_B and fc9_SH        
        Concat_BSH = torch.cat([fc2_B, fc9_SH], 1)  # 1024 + 1024        
        cls_score_BSH = self.cls_score_BSH(Concat_BSH)
        cls_prob_BSH  = torch.sigmoid(cls_score_BSH)
        self.predictions["cls_score_BSH"]  = cls_score_BSH
        self.predictions["cls_prob_BSH"]   = cls_prob_BSH

        return True  

    def add_loss_2D_branch(self, debug=False):
        # here use cls_score, not cls_prob
        cls_score_H      = self.predictions["cls_score_H"]
        cls_score_O      = self.predictions["cls_score_O"]
        cls_score_sp     = self.predictions["cls_score_sp"]
        
        cls_score_H_with_weight   = cls_score_H.mul(self.HO_weight)
        cls_score_O_with_weight   = cls_score_O.mul(self.HO_weight)
        cls_score_sp_with_weight  = cls_score_sp.mul(self.HO_weight)

        label_HO              = self.gt_class_HO.squeeze(0)
        H_cross_entropy       = self.BCEcriterion(cls_score_H_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        O_cross_entropy       = self.BCEcriterion(cls_score_O_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        sp_cross_entropy      = self.BCEcriterion(cls_score_sp_with_weight, label_HO)

        # 2D branch
        loss = 0
        for lossTerm in self.cfg.BRANCH_2D_LOSS_TERMS:
            if lossTerm == 'sp':
                loss += sp_cross_entropy
            elif lossTerm == 'H':
                loss += H_cross_entropy
            elif lossTerm == 'O':
                loss += O_cross_entropy
        
        self.losses['total_loss'] = loss

        return loss

    '''
        Joint Loss (2D + 3D branch)
    '''
    def add_loss(self, debug=False):
        # here use cls_score, not cls_prob
        cls_score_H  = self.predictions["cls_score_H"]
        cls_score_O  = self.predictions["cls_score_O"]
        cls_score_sp = self.predictions["cls_score_sp"]
        cls_score_D  = self.predictions['cls_score_D']
        cls_score_J  = self.predictions['cls_score_J']
        
        cls_score_H_with_weight   = cls_score_H.mul(self.HO_weight)
        cls_score_O_with_weight   = cls_score_O.mul(self.HO_weight)
        cls_score_sp_with_weight  = cls_score_sp.mul(self.HO_weight)
        cls_score_D_with_weight   = cls_score_D.mul(self.HO_weight)
        cls_score_J_with_weight   = cls_score_J.mul(self.HO_weight)

        label_HO         = self.gt_class_HO.squeeze(0)
        H_cross_entropy  = self.BCEcriterion(cls_score_H_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        O_cross_entropy  = self.BCEcriterion(cls_score_O_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        sp_cross_entropy = self.BCEcriterion(cls_score_sp_with_weight, label_HO)
        D_cross_entropy  = self.BCEcriterion(cls_score_D_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        J_cross_entropy  = self.BCEcriterion(cls_score_J_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        
        L_cls = H_cross_entropy + O_cross_entropy + D_cross_entropy + J_cross_entropy + sp_cross_entropy

        cls_prob_R = self.predictions['cls_prob_R']
        cls_prob_D = self.predictions['cls_prob_D']
        bias  = cls_prob_R - cls_prob_D
        L_sem = torch.mean(bias * bias)        
        L_tri = self.losses['L_tri']
        A_2D  = self.predictions['A_2D']
        A_3D  = self.predictions['A_3D']

        if self.cfg.ATTENTION_LOSS == 'KL':
            # KullbackLeibler divergence
            L_att = torch.mean(A_2D * torch.log(F.relu(A_2D/A_3D) + 1e-5))
            
        elif self.cfg.ATTENTION_LOSS == 'MSE':
            # MSE Loss
            L_att = nn.MSELoss()(A_2D, A_3D)
        
        if torch.isnan(L_att):
            L_att = 0 

        self.losses['L_cls/H_cross_entropy']  = H_cross_entropy
        self.losses['L_cls/O_cross_entropy']  = O_cross_entropy
        self.losses['L_cls/D_cross_entropy']  = D_cross_entropy
        self.losses['L_cls/J_cross_entropy']  = J_cross_entropy
        self.losses['L_cls/sp_cross_entropy'] = sp_cross_entropy
        self.losses['L_att']                  = L_att
        self.losses['L_sem']                  = L_sem
            
        # print(L_cls, L_sem, L_tri, L_att)
        loss = L_cls + 0.01 * L_sem + 0.001 * L_tri + 0.00001 * L_att
        self.losses['total_loss'] = loss

        return loss
    

    # loss function for the new classification brance
    def add_loss_BSH_branch(self, debug=False):
        # here use cls_score, not cls_prob
        cls_score_BSH               = self.predictions["cls_score_BSH"]
        cls_score_BSH_with_weight   = cls_score_BSH.mul(self.HO_weight)        
        label_HO                    = self.gt_class_HO.squeeze(0)
        BSH_cross_entropy           = self.BCEcriterion(cls_score_BSH_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        loss = BSH_cross_entropy        
        self.losses['total_loss'] = loss

        return loss


'''
test script : Run testcases to check dimension
'''
if __name__ == "__main__":
    print("[Test Mode] ==> Building DJR_ResNet50")
    net = DJR_ResNet50()
    print("[Test Mode] ==> Build DJR_ResNet50 successfully")

    # test sp_to_head
    data_input = torch.randn(10, 2, 64, 64)
    output = net.sp_to_head(data_input)
    print(output.size())
    # assert()
