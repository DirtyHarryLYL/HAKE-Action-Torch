# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of detector"""
from abc import ABC, abstractmethod


def get_detector(detector_mode='yolo', opt=None):
    if detector_mode == 'yolo':
        from .yolo_api import YOLODetector
        from .yolo_cfg import cfg
        cfg.CONFIG = opt.yolo_cfg
        cfg.WEIGHTS = opt.yolo_weight
        return YOLODetector(cfg, opt)
    elif detector_mode == 'tracker':
        from .tracker_api import Tracker
        from .tracker_cfg import cfg
        cfg.CONFIG = opt.yolo_cfg
        cfg.WEIGHTS = opt.yolo_weight
        return Tracker(cfg, opt)
    elif detector_mode.startswith('efficientdet_d'):
        from .effdet_api import EffDetDetector
        from .effdet_cfg import cfg
        return EffDetDetector(cfg, opt)
    else:
        raise NotImplementedError


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def images_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass
