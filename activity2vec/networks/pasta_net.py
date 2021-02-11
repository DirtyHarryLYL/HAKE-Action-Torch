##################################################################################
#  Author: Hongwei Fan                                                           #
#  E-mail: hwnorm@outlook.com                                                    #
#  Homepage: https://github.com/hwfan                                            #
#  Based on PaStaNet in CVPR'20                                                  #
#  TF version:                                                                   #
#  https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action  #
##################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .resnet_v1.resnetv1_torch import resnet50 as resnet50_v1

class pasta_res50(nn.Module):

    def __init__(self, cfg):
        super(pasta_res50, self).__init__()
        
        self.cfg             = cfg
        self.num_verbs       = cfg.DATA.NUM_VERBS
        self.num_parts       = cfg.DATA.NUM_PARTS
        self.pasta_idx2name  = cfg.DATA.PASTA_NAMES

        self.pasta_name2idx  = dict()
        self.num_pastas      = []
        for pasta_idx, part_name in enumerate(self.pasta_idx2name):
            self.pasta_name2idx[part_name] = pasta_idx
            self.num_pastas.append(cfg.DATA.NUM_PASTAS[part_name.upper()])
        
        self.num_fc          = cfg.MODEL.NUM_FC
        self.scene_dim       = 1024
        self.human_dim       = 2048
        self.roi_dim         = 1024
        self.part_agg_rule   = cfg.MODEL.PART_AGG_RULE
        self.part_agg_num    = [len(part_agg_rule) for part_agg_rule in self.part_agg_rule]
        if self.cfg.MODEL.PART_ROI_ENABLE:
            self.num_fc_parts  = [part_agg_num*self.roi_dim + self.scene_dim + self.human_dim for part_agg_num in self.part_agg_num]
        else:
            self.num_fc_parts  = [self.scene_dim + self.human_dim for part_agg_num in self.part_agg_num]
        
        if self.cfg.MODEL.POSE_MAP:
            self.num_fc_parts  = [(x + cfg.MODEL.SKELETON_DIM) for x in self.num_fc_parts]
            
        self.module_trained = cfg.MODEL.MODULE_TRAINED
        self.dropout_rate   = cfg.MODEL.DROPOUT
        self.pasta_language_matrix  = torch.from_numpy(np.load(cfg.DATA.PASTA_LANGUAGE_MATRIX_PATH)).cuda()
        self.resnet50 = resnet50_v1()
        self.resnet50.conv1.padding = 0

        ########################
        # Building the network #
        ########################

        # ResNet-style image head.
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

        # Human feature extractor.
        self.resnet_layer4 = self.resnet50.layer4

        # PaSta classifier.
        self.fc7_parts   = nn.ModuleList(
                                            [
                                                nn.Sequential(
                                                    nn.Linear(self.num_fc_parts[pasta_idx], self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(self.dropout_rate),
                                                    nn.Linear(self.num_fc, self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(self.dropout_rate)
                                                ) 
                                                for pasta_idx in range(len(self.pasta_idx2name))
                                            ]
                                        )

        self.part_cls_scores = nn.ModuleList(
                                                [
                                                    nn.Linear(self.num_fc, self.num_pastas[pasta_idx])
                                                    for pasta_idx in range(len(self.pasta_idx2name))
                                                ]
                                            )
        
        # Verb classifier.
        if cfg.MODEL.VERB_ONE_MORE_FC:
            self.verb_cls_scores = nn.Sequential(
                            nn.Linear(len(self.pasta_idx2name) * self.num_fc, self.num_fc),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.num_fc, self.num_verbs)
                        ) 
        else:
            self.verb_cls_scores = nn.Linear(len(self.pasta_idx2name) * self.num_fc, self.num_verbs)

        ##############################
        # Freeze the useless params. #
        ##############################

        if cfg.TRAIN.FREEZE_BACKBONE:
            for p in self.image_to_head.parameters():
                p.requires_grad = False
            if cfg.TRAIN.FREEZE_RES4:
                for p in self.resnet_layer4.parameters():
                    p.requires_grad = False
            
        for pasta_idx in range(len(self.pasta_idx2name)):
            for p in self.fc7_parts[pasta_idx].parameters():
                p.requires_grad = self.pasta_idx2name[pasta_idx] in self.module_trained
            for p in self.part_cls_scores[pasta_idx].parameters():
                p.requires_grad = self.pasta_idx2name[pasta_idx] in self.module_trained

        for p in self.verb_cls_scores.parameters():
            p.requires_grad = 'verb' in self.module_trained

        ###############################################
        # Building the extractor of pose map feature. #
        ###############################################

        if cfg.MODEL.POSE_MAP:
            self.pool2_flat_pose_maps = nn.ModuleList(
                                                [
                                                    nn.Sequential(nn.Conv2d(1, 32, (5, 5), stride=(1, 1), padding=0),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.MaxPool2d((2, 2)),

                                                                  nn.Conv2d(32, 16, (5, 5), stride=(1, 1), padding=0),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.MaxPool2d((2, 2)))
                                                    for pasta_idx in range(len(self.pasta_idx2name))
                                                ]
                                            )
            for pasta_idx in range(len(self.pasta_idx2name)):
                for p in self.pool2_flat_pose_maps[pasta_idx].parameters():
                    p.requires_grad = self.pasta_idx2name[pasta_idx] in self.module_trained
                    

    def _crop_pool_layer(self, bottom, rois, max_pool=False):
        '''
        @input:
            bottom: [1, 1024, H, W]
            rois  : [1, N, 5]
        @output: 
            crops : [N, 1024, 7, 7]
        '''
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
            pre_pool_size = self.cfg.POOLING_SIZE * 2
            grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)), align_corners=False)

            all_roi = []
            for j in range(rois.size(0)):
                _grid = grid.narrow(0, j, 1)
                _roi_feature = F.grid_sample(bottom.view(1,bottom.size(1), bottom.size(2), bottom.size(3)), _grid, align_corners=False)
                all_roi.append(_roi_feature)
            crops = torch.cat(all_roi)
            # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
            crops = F.max_pool2d(crops, 2, 2)
        else:
            grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, self.cfg.POOLING_SIZE, self.cfg.POOLING_SIZE)), align_corners=False)

            all_roi = []
            for j in range(rois.size(0)):
                _grid = grid.narrow(0, j, 1)
                _roi_feature = F.grid_sample(bottom.view(1,bottom.size(1), bottom.size(2), bottom.size(3)), _grid, align_corners=False)
                all_roi.append(_roi_feature)
            crops = torch.cat(all_roi)
            # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)

        return crops

    # image/frame --> resnet --> part RoI features + pose map feature --> PaSta (Part States) recognition --> verb (whole body action) recognition
    def forward(self, image, annos):
        # Extract the feature of skeleton image.
        if self.cfg.MODEL.POSE_MAP:
            skeleton_feats = []
            for pasta_idx in range(len(self.pasta_idx2name)):
                skeleton_feat = self.pool2_flat_pose_maps[pasta_idx](annos['skeletons'])
                skeleton_feat = skeleton_feat.view(skeleton_feat.shape[0], -1)
                skeleton_feats.append(skeleton_feat)

        head = self.image_to_head(image)

        # scene/context (whole image) feature
        f_scene = torch.mean(head, [2, 3])

        # human roi feature
        f_human_roi = self._crop_pool_layer(head, annos['human_bboxes'])
        f_human = self.resnet_layer4(f_human_roi)
        f_human = torch.mean(f_human, [2, 3])
        
        #############################################################################################
        #  To simplify the model, the interacted object feature is not leveraged here.              #        
        #  You could also utilize object to further enhance the PaSta and verb recognition for HOI. #
        #  The object boxes can be obtained from your own detectors based on COCO, LVIS, FSOD, etc. #
        #  More details about HOI detection please refer to our survey repo:                        #
        #  https://github.com/DirtyHarryLYL/HOI-Learning-List                                       #
        #############################################################################################
        # object roi feature
#         f_object_roi = self._crop_pool_layer(head, annos['object_bboxes']) # detected boxes from detectors
#         f_object = self.resnet_layer4(f_object_roi)
#         f_object = torch.mean(f_object, [2, 3])

        # part roi feature
        if self.cfg.MODEL.PART_ROI_ENABLE:
            f_parts_roi = []
            for part_idx in range(self.num_parts):
                f_part_roi = torch.mean(self._crop_pool_layer(head, annos['part_bboxes'][:, part_idx, :]), [2, 3])
                f_parts_roi.append(f_part_roi)
            
            f_scene_for_part = f_scene.repeat([f_parts_roi[0].shape[0], 1])
            f_base = [f_human, f_scene_for_part]
            f_parts_agg = []
            for part_agg_rule in self.part_agg_rule:
                f_part = [f_parts_roi[part_idx] for part_idx in part_agg_rule]
                f_part = f_part + f_base
                f_part = torch.cat(f_part, 1)
                f_parts_agg.append(f_part)
        else:
            f_scene_for_part = f_scene.repeat([f_human.shape[0], 1])
            f_base = torch.cat([f_human, f_scene_for_part], 1)
            f_parts_agg = [f_base for pasta_idx in range(len(self.cfg.DATA.PASTA_NAMES))]
            
        f_parts = []
        s_parts = []
        p_parts = []

        # classify the part states
        for part_idx, f_part in enumerate(f_parts_agg):
            if self.cfg.MODEL.POSE_MAP:
                f_part_cat  = torch.cat([f_part, skeleton_feats[part_idx]], 1)
                f_part_fc7  = self.fc7_parts[part_idx](f_part_cat)
            else:
                f_part_fc7  = self.fc7_parts[part_idx](f_part)
                
            s_part  = self.part_cls_scores[part_idx](f_part_fc7)
            p_part  = torch.sigmoid(s_part)
            f_parts.append(f_part_fc7)
            s_parts.append(s_part)
            p_parts.append(p_part)
        
        f_pasta_visual = torch.cat(f_parts, 1)
        p_pasta = torch.cat(p_parts, 1)

        # classify the verbs
        s_verb = self.verb_cls_scores(f_pasta_visual)
        p_verb = torch.sigmoid(s_verb)

        f_pasta_language = torch.matmul(p_pasta, self.pasta_language_matrix)
        f_pasta = torch.cat([f_pasta_visual, f_pasta_language], 1)

        # return the pasta feature and pasta probs if in test/inference mode, 
        # else return the pasta scores for loss input.
        
        if not self.training:
            return f_pasta, p_pasta, p_verb
        else:
            return s_parts, s_verb
