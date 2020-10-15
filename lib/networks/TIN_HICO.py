import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import pickle
import h5py
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ult.config_TIN  import cfg
from ult.roi_pooling import ROIPooling2dPytorch as roi_pooling
from ult.ult import write_hdf5
from .resnet_v1.resnetv1_torch import resnet50 as resnet50_v1
from .resnet_v1.resnetv1_torch import resnetv1

'''
@input:
    name:           shape:
    image           [1, None, None, 3]
    spatial         [None, 64, 64, 3]
    H_boxes         [None, 5]
    O_boxes         [None, 5]
    gt_class_HO     [None, 600]
    gt_binary_label [None, 2]
'''
class TIN_ResNet50(nn.Module):
    def __init__(self):
        super(TIN_ResNet50, self).__init__()

        '''
        global settings
        '''
        self.num_classes = 600  # num of HOI classes
        self.num_fc = 1024
        self.num_binary = 2

        # image_to_head : feature extractor
        # self.resnet50 = torch.hub.load('pytorch/vision:v0.4.0', 'resnet50', pretrained=True)
        self.resnet50 = resnet50_v1()
        self.resnet50.layer5 = copy.deepcopy(self.resnet50.layer4)

        # freeze layers
        def freeze(m):
            m.requires_grad=False
            
        self.resnet50.conv1.apply(freeze)
        self.resnet50.bn1.apply(freeze)
        self.resnet50.layer1.apply(freeze)
        self.resnet50.conv1.padding = 0

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
        # self.resnet_layer5   = self.resnet50.layer5
        self.resnet_layer5   = copy.deepcopy(self.resnet50.layer4)

        self.crop_pool_layer = roi_pooling(output_size=(7,7), spatial_scale=float(1/16))

        '''
        spatial configuration, conv-pool-conv-pool-flatten
        input:  [num_pos_neg, 2, 64, 64]
        output: [num_pos_neg, 5408]
        '''
        self.sp_to_head = nn.Sequential(
            nn.Conv2d(2, 64, (5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0),

            nn.Conv2d(64, 32, (5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0),

            # nn.Flatten(),
        )

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

        self.head_bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, (1, 1)),
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
            nn.Linear(7456, self.num_fc, bias=False),
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
            nn.MaxPool2d((2, 2)),

            # nn.Flatten()
        )

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

        self.cls_score_H      = nn.Linear(1024, self.num_classes)
        self.cls_score_O      = nn.Linear(1024, self.num_classes)
        self.cls_score_sp     = nn.Linear(1024, self.num_classes)
        self.cls_score_binary = nn.Linear(1024, self.num_binary)

        self.predictions = {}
        self.visualize   = {}
        self.losses      = {}
        self.HO_weight   = torch.tensor([
                9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423, 
                11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699, 
                6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912, 
                5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048, 
                8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585, 
                12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731, 
                7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116, 
                14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678, 
                12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933, 
                14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973, 
                12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967, 
                4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679, 
                9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917, 
                10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057, 
                8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799, 
                9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912, 
                9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591, 
                12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264, 
                11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264, 
                7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304, 
                10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384, 
                11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143, 
                11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264, 
                7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264, 
                8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909, 
                7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748, 
                14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135, 
                11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411, 
                14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533, 
                10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361, 
                9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222, 
                9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963, 
                8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324, 
                9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948, 
                5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264, 
                11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248, 
                10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862, 
                8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224, 
                12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973, 
                12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493, 
                14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973, 
                11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892, 
                10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825, 
                12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799, 
                9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505, 
                12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368, 
                7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925, 
                7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015, 
                7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551, 
                9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584, 
                5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826, 
                11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799, 
                10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059, 
                13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571, 
                11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501, 
                14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411, 
                7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305, 
                11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186, 
                12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264, 
                10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
            ]).cuda()
        self.binary_weight = torch.tensor([1.6094379124341003, 0.22314355131420976]).cuda()

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
        fc9_SH = self.fc9_SH(fc8_SH) # OK


        Concat_SO = torch.cat([fc7_O, fc7_SO], 1)

        if debug:
            print("==> [head_to_tail] fc7_O.shape: ", fc7_O.shape)
            print("==> [head_to_tail] fc7_SO.shape: ", fc7_SO.shape)
            print("==> [head_to_tail] Concat_SO.shape: ", Concat_SO.shape)

        fc8_SO = self.fc8_SO(Concat_SO)
        fc9_SO = self.fc9_SO(fc8_SO)
        Concat_SHsp = torch.cat([fc7_H, sp[:int(fc7_H.shape[0])]], 1)

        if debug:
            print("==> [head_to_tail] sp.shape: ", sp.shape)
            print("==> [head_to_tail] Concat_SHsp.shape: ", Concat_SHsp.shape)

        Concat_SHsp = self.Concat_SHsp(Concat_SHsp)
        fc7_SHsp = self.fc7_SHsp(Concat_SHsp) # OK

        return fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO

    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, spatial, debug=False):
        pool2_flat_pose_map = self.pool2_flat_pose_map(spatial)
        pool2_flat_pose_map = torch.flatten(pool2_flat_pose_map.permute(0,2,3,1), start_dim=1)

        fc_binary_1 = torch.cat([fc7_H, fc7_SH], 1)

        if debug:
            print("==> [binary_discriminator] spatial.shape: ", spatial.shape) # 
            print("==> [binary_discriminator] pool2_flat_pose_map.shape: ", pool2_flat_pose_map.shape) # N, 2704
            print("==> [binary_discriminator] fc_binary_1.shape: ", fc_binary_1.shape) # [pos + neg, 8480]

        fc_binary_1 = torch.cat([fc_binary_1, sp, pool2_flat_pose_map], 1)
        fc8_binary_1 = self.fc8_binary_1(fc_binary_1)
        
        fc_binary_2 = torch.cat([fc7_O, fc7_SO], 1)
        fc8_binary_2 = self.fc8_binary_2(fc_binary_2)
    
        # remove concat here (by self.H_num)

        fc8_binary = torch.cat([fc8_binary_1, fc8_binary_2], 1)
        fc9_binary = self.fc9_binary(fc8_binary)

        if debug:
            print("==> [binary_discriminator] pool2_flat_pose_map.shape: ", pool2_flat_pose_map.shape) # N, 2704
            print("==> [binary_discriminator] fc_binary_1.shape: ", fc_binary_1.shape) # [pos + neg, 8480]
            print("==> [binary_discriminator] fc8_binary_1.shape: ", fc8_binary_1.shape) # 
            print("==> [binary_discriminator] fc_binary_2.shape: ", fc_binary_2.shape) # [pos, 3072]
            # print("==> [binary_discriminator] 2 index = {}, self.H_num = {}, fc8_binary_2.shape = {}".format(index, self.H_num, fc8_binary_2.shape)) # [pos + neg, 1024]
            print("==> [binary_discriminator] fc8_binary.shape: ", fc8_binary.shape) # 

        return fc9_binary

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, debug=False):
        cls_score_H = self.cls_score_H(fc7_H)
        cls_prob_H  = torch.sigmoid(cls_score_H)   
        # torch.reshape(cls_prob_H, (1, self.num_classes))

        cls_score_O = self.cls_score_O(fc7_O)
        cls_prob_O  = torch.sigmoid(cls_score_O)   
        # torch.reshape(cls_prob_O, (1, self.num_classes))

        cls_score_sp = self.cls_score_sp(fc7_SHsp)
        cls_prob_sp = torch.sigmoid(cls_score_sp)
        
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
        self.predictions["cls_prob_HO"] = cls_prob_sp * \
            (cls_prob_O + cls_prob_H)

        return cls_prob_H, cls_prob_O, cls_prob_sp

    def binary_classification(self, fc9_binary):
        cls_score_binary = self.cls_score_binary(fc9_binary)
        cls_prob_binary = torch.sigmoid(cls_score_binary)
        # torch.reshape(cls_prob_binary, (1, self.num_binary))

        self.predictions["cls_score_binary"] = cls_score_binary
        self.predictions["cls_prob_binary"]  = cls_prob_binary

        return cls_prob_binary

    def forward(self, blobs, debug=False):
        image   = blobs['image']
        H_boxes = blobs['H_boxes']
        O_boxes = blobs['O_boxes']
        spatial = blobs['sp']
        self.H_num = int(blobs['H_num'])

        if not self.testMode:
            self.gt_class_HO = blobs['gt_class_HO']
            self.gt_binary_label = blobs['binary_label']

        head = self.image_to_head(image)

        try:
            # try crop and pool
            if self.testMode:
                pool5_H = self._crop_pool_layer(head, H_boxes)
                pool5_O = self._crop_pool_layer(head, O_boxes)
            else:
                pool5_H = self._crop_pool_layer(head, H_boxes.squeeze(0))
                pool5_O = self._crop_pool_layer(head, O_boxes.squeeze(0))
        except:
            # crop fail
            if self.testMode:
                return torch.tensor([-1]), torch.tensor([-1])
            else:
                return False

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O)

        # whole image feature
        head_phi = self.head_phi(head)
        head_g   = self.head_g(head)

        Att_H = self.attention_pool_layer_H(head_phi, fc7_H)

        # norm H
        Att_H_shape = Att_H.shape
        Att_H = Att_H.view(Att_H_shape[0], Att_H_shape[1], -1)
        Att_H = Att_H.softmax(-1)
        Att_H = Att_H.view(Att_H_shape)
        att_head_H = head_g.mul(Att_H)

        Att_O = self.attention_pool_layer_O(head_phi, fc7_O)

        # norm O
        Att_O_shape = Att_O.shape
        Att_O = Att_O.view(Att_O_shape[0], Att_O_shape[1], -1)
        Att_O = Att_O.softmax(-1)
        Att_O = Att_O.view(Att_O_shape)
        att_head_O = head_g.mul(Att_O)

        pool5_SH = self.head_bottleneck(att_head_H)
        pool5_SO = self.head_bottleneck(att_head_O)

        
        if self.testMode:
            sp = self.sp_to_head(spatial[:,:,:,0:2].permute(0,3,1,2))
            # sp = self.sp_to_head(spatial[0,:,:,:,0:2].permute(0,3,1,2))
            sp = torch.flatten(sp.permute(0,2,3,1), start_dim=1)
        else:
            sp = self.sp_to_head(spatial[0,:,:,:,0:2].permute(0,3,1,2))
            sp = torch.flatten(sp.permute(0,2,3,1), start_dim=1)

        fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO = self.head_to_tail(
            fc7_H, fc7_O, pool5_SH, pool5_SO, sp)

        if self.testMode:
            fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, spatial[:,:,:,2:].permute(0,3,1,2))
        else:
            fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, spatial[0,:,:,:,2:].permute(0,3,1,2))

        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(
            fc9_SH, fc9_SO, fc7_SHsp)

        # add a Discriminator here to make binary classification
        cls_prob_binary = self.binary_classification(fc9_binary)
        
        self.visualize["attention_map_H"] = (
            Att_H - torch.min(Att_H[0, :, :, :])) / torch.max((Att_H[0, :, :, :] - torch.min(Att_H[0, :, :, :])))
        self.visualize["attention_map_O"] = (
            Att_O - torch.min(Att_O[0, :, :, :])) / torch.max((Att_O[0, :, :, :] - torch.min(Att_O[0, :, :, :])))

        if self.testMode:
            # TEST script 1: test_HICO_ori.py
            # return self.predictions["cls_prob_HO"], self.predictions["cls_score_binary"]

            # TEST script 2: test_HICO.py
            return self.predictions["cls_prob_HO"], cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary
        else:
            # train mode, return True
            return True

    def add_loss(self, debug=False):
        # here use cls_score, not cls_prob
        cls_score_binary = self.predictions["cls_score_binary"]
        cls_score_H      = self.predictions["cls_score_H"]
        cls_score_O      = self.predictions["cls_score_O"]
        cls_score_sp     = self.predictions["cls_score_sp"]
        
        cls_score_H_with_weight      = cls_score_H.mul(self.HO_weight)
        cls_score_O_with_weight      = cls_score_O.mul(self.HO_weight)
        cls_score_sp_with_weight     = cls_score_sp.mul(self.HO_weight)
        cls_score_binary_with_weight = cls_score_binary.mul(self.binary_weight)

        label_HO     = self.gt_class_HO.squeeze(0)
        label_binary = self.gt_binary_label.squeeze(0)

        if debug:
            print("==> [add_loss] label_HO.shape = {}, label_binary.shape = {}".format(label_HO.shape, label_binary.shape))
            print("==> [add_loss] cls_score_binary_with_weight.shape = {}".format(cls_score_binary_with_weight.shape))

        binary_cross_entropy  = self.BCEcriterion(cls_score_binary_with_weight, label_binary)
        H_cross_entropy       = self.BCEcriterion(cls_score_H_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        O_cross_entropy       = self.BCEcriterion(cls_score_O_with_weight[:self.H_num, :], label_HO[:self.H_num, :])
        sp_cross_entropy      = self.BCEcriterion(cls_score_sp_with_weight, label_HO)

        self.losses['binary_cross_entropy'] = binary_cross_entropy
        self.losses['H_cross_entropy']      = H_cross_entropy
        self.losses['O_cross_entropy']      = O_cross_entropy
        self.losses['sp_cross_entropy']     = sp_cross_entropy

        if cfg.TRAIN_MODULE == 1:
            loss = H_cross_entropy + O_cross_entropy + \
                    sp_cross_entropy + binary_cross_entropy
        elif cfg.TRAIN_MODULE == 2:
            loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy
        elif cfg.TRAIN_MODULE == 3:
            loss = binary_cross_entropy
        elif cfg.TRAIN_MODULE == 4:
            loss = H_cross_entropy + O_cross_entropy
        else:
            loss = sp_cross_entropy

        # loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy + binary_cross_entropy
        self.losses['total_loss'] = loss

        return loss

'''
# test script : Run testcases to check dimension
if __name__ == "__main__":
    print("[Test Mode] ==> Building TIN_ResNet50")
    net = TIN_ResNet50()
    print("[Test Mode] ==> Build TIN_ResNet50 successfully")

    # test sp_to_head
    data_input = torch.randn(10, 2, 64, 64)
    output = net.sp_to_head(data_input)
    print(output.size())
    # assert()
'''
