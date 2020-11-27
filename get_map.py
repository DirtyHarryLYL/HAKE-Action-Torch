import pickle
import numpy as np
import os
import argparse
import h5py
from HICO_DET_utils import rare, obj_range, calc_ap_ko

def parse_arg():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--exp', dest='exp',
            help='Define exp name',
            default='_'.join(time.asctime(time.localtime(time.time())).split()), type=str)
    args = parser.parse_args()
    return args

args = parse_arg()
model  = args.exp

result_file = 'exp/' + model + '/result.pkl'
res         = pickle.load(open(result_file, 'rb'))
keys   = res['keys']
bboxes = res['bboxes']
scores = res['scores']
hdet   = res['hdet']
odet   = res['odet']
sel    = res['sel']
map, map_ko = np.zeros(600), np.zeros(600)
mrec, mrec_ko = np.zeros(600), np.zeros(600)

for i in range(80):
    if len(keys[i]) == 0:
        continue
    begin = obj_range[i][0] - 1
    end   = obj_range[i][1]
    ko_mask = []
    for hoi_id in range(begin, end):
        gt_bbox = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'))
        ko_mask += list(gt_bbox.keys())
    ko_mask = set(ko_mask)
    
    for hoi_id in range(begin, end):
        select = sel[hoi_id]
        bbox   = bboxes[i][select, :]
        key    = keys[i][select]
        score  = scores[i][select, :]
        map[hoi_id], mrec[hoi_id], map_ko[hoi_id], mrec_ko[hoi_id] = calc_ap_ko(score, bbox, key, hoi_id, begin, ko_mask)

print('eval mode: default\n')
print('total    ap: %.4f rec: %.4f \n' % (float(np.mean(map)),float(np.mean(mrec))))
print('rare     ap: %.4f rec: %.4f \n' % (float(np.mean(map[rare > 1])), float(np.mean(mrec[rare > 1]))))
print('non-rare ap: %.4f rec: %.4f \n' % (float(np.mean(map[rare < 1])), float(np.mean(mrec[rare < 1]))))
print
print('eval mode: known object\n')
print('total    ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko)), float(np.mean(mrec_ko))))
print('rare     ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko[rare > 1])), float(np.mean(mrec_ko[rare > 1]))))
print('non-rare ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko[rare < 1])), float(np.mean(mrec_ko[rare < 1]))))

with open('exp/'+model+'/eval_result.txt', 'w') as f:
    f.write('eval mode: default\n')
    f.write('total    ap: %.4f rec: %.4f \n' % (float(np.mean(map)),float(np.mean(mrec))))
    f.write('rare     ap: %.4f rec: %.4f \n' % (float(np.mean(map[rare > 1])), float(np.mean(mrec[rare > 1]))))
    f.write('non-rare ap: %.4f rec: %.4f \n' % (float(np.mean(map[rare < 1])), float(np.mean(mrec[rare < 1]))))

    f.write('eval mode: known object\n')
    f.write('total    ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko)), float(np.mean(mrec_ko))))
    f.write('rare     ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko[rare > 1])), float(np.mean(mrec_ko[rare > 1]))))
    f.write('non-rare ap: %.4f rec: %.4f \n' % (float(np.mean(map_ko[rare < 1])), float(np.mean(mrec_ko[rare < 1]))))
