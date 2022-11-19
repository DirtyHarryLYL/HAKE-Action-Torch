from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from datasets import hake_meta


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PaStaHead(nn.Module):
    def __init__(self,
                 in_channels,
                 loss_weight=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.loss_weight = loss_weight
        self.focal_alpha = 3
        self.focal_gamma = 1

        self._init_layers(in_channels)

    def _init_layers(self, in_channels):

        self.verb_fc = MLP(in_channels, in_channels, hake_meta.num_classes['verb'], 3)
        self.foot_fc = MLP(in_channels, in_channels, hake_meta.num_classes['foot'], 3)
        self.leg_fc = MLP(in_channels, in_channels, hake_meta.num_classes['leg'], 3)
        self.hip_fc = MLP(in_channels, in_channels, hake_meta.num_classes['hip'], 3)
        self.hand_fc = MLP(in_channels, in_channels, hake_meta.num_classes['hand'], 3)
        self.arm_fc = MLP(in_channels, in_channels, hake_meta.num_classes['arm'], 3)
        self.head_fc = MLP(in_channels, in_channels, hake_meta.num_classes['head'], 3)
        self.pasta_binary_fc = MLP(in_channels, in_channels, hake_meta.num_classes['pasta_binary'], 3)

        self.train_set = {
            'verb': self.verb_fc,
            'foot': self.foot_fc,
            'leg': self.leg_fc,
            'hip': self.hip_fc,
            'hand': self.hand_fc,
            'arm': self.arm_fc,
            'head': self.head_fc,
            'pasta_binary': self.pasta_binary_fc
        }

    def forward_train(self,
                      x,
                      gt_labels,
                      img_metas,
                      **kwargs):

        cls_scores = self.predict(x)

        losses = dict()
        for i, name in enumerate(self.train_set.keys()):
            gt_label = gt_labels[name].to(x.device)
            cls_score = cls_scores[i]
            loss = self.loss(cls_score, gt_label, name)
            losses.update(loss)
        return losses

    def predict(self, x):
        cls_scores = []
        for k, fc in self.train_set.items():
            cls_score = fc(x)
            cls_scores.append(cls_score)
        return cls_scores

    def loss(self, logit, label, name):
        loss = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        pt = torch.exp(-loss)
        loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
        loss = loss.mean()
        loss = loss * self.loss_weight

        pred = (logit.detach().sigmoid() > 0.5)
        correct = (pred == label).sum()
        acc = 100. * correct / pred.numel()

        out = {
            f'loss_{name}': loss,
            f'accuray_{name}': acc
        }
        return out

    def simple_test(self, x, gt_labels, img_metas, **kwargs):
        assert x.size(0) == 1

        results = {'prob': OrderedDict(), 'label': OrderedDict()}
        cls_scores = self.predict(x)

        for i, name in enumerate(self.train_set.keys()):
            prob = cls_scores[i].sigmoid().view(-1)
            label = gt_labels[name].view(-1)
            results['prob'][name] = prob.cpu().numpy()
            results['label'][name] = label.cpu().numpy()

        return [results]

    def init_weights(self):
        pass


class PaStaNet(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, gt_labels, img_metas):
        feat = self.backbone(x)
        if self.training:
            loss = self.head.forward_train(feat, gt_labels, img_metas)
            return loss
        else:
            pred = self.head.simple_test(feat, gt_labels, img_metas)
            return pred


def build(args):
    backbone = build_backbone(args)
    head = PaStaHead(backbone.out_dim)
    model = PaStaNet(backbone, head)
    return model
