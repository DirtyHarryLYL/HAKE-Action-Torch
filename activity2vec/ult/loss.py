# Copied from fvcore/nn/focal_loss.py of fvcore repository: 
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn import functional as F


def sigmoid_focal_loss(
    inputs,
    targets,
    weight,
    alpha=-1,
    gamma=2,
    reduction="none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def a2v_loss(cfg, s_parts, s_verb, annos, pasta_weights, pasta_name2idx):
    losses = []
    if cfg.TRAIN.LOSS_TYPE == 'bce':
        loss_func = F.binary_cross_entropy_with_logits
    elif cfg.TRAIN.LOSS_TYPE == 'focal':
        loss_func = sigmoid_focal_loss
    else:
        raise NotImplementedError

    for module_name in cfg.MODEL.MODULE_TRAINED:
        if module_name != 'verb':
            pasta_idx = pasta_name2idx[module_name]
            preds = s_parts[pasta_idx]
            labels = annos['pasta'][module_name]
            weight = pasta_weights[module_name].repeat(s_parts[pasta_idx].shape[0], 1)
        else:
            preds = s_verb
            labels = annos['verbs']
            weight = pasta_weights[module_name].repeat(s_verb.shape[0], 1)
        
        loss = loss_func(preds, 
                         labels, 
                         weight=weight if cfg.TRAIN.WITH_LOSS_WTS else None, 
                         reduction='mean'
                         )
        losses.append(loss)

    return sum(losses)