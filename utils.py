import time
import numpy as np
import os.path as osp
import torch
args      = pickle.load(open('arguments.pkl', 'rb'))
HO_weight = torch.from_numpy(args['HO_weight'])
fac_i     = args['fac_i']
fac_a     = args['fac_a']
fac_d     = args['fac_d']
nis_thresh= args['nis_thresh']

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.cnt = 0

    def update(self, val, k):
        self.avg = self.avg + (val - self.avg) * k / (self.cnt + k)
        self.cnt += k

    def __str__(self):
        return '%.4f' % self.avg
