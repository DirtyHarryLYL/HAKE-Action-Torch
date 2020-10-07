import pickle
import numpy as np
from HICO_DET_utils import rare, obj_range, calc_ap_ko
import pickle 

score, bboxes, keys, sel = pickle.load(open('supp_res.pkl', 'rb'))

summary_1 = {
    'ap': np.zeros(600),
    'rec': np.zeros(600),
    'ap_ko': np.zeros(600),
    'rec_ko': np.zeros(600),
}

for obj_index in range(80):
    x, y = obj_range[obj_index]
    x -= 1
    ko_mask = []
    for hoi_id in range(x, y):
        gt_bbox = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'), encoding='latin1')
        ko_mask += list(gt_bbox.keys())
    ko_mask = set(ko_mask)
    for hoi_id in range(x, y):
        output_1 = calc_ap_ko(score[obj_index][sel[hoi_id]], bboxes[obj_index][sel[hoi_id]], keys[obj_index][sel[hoi_id]], hoi_id, x, ko_mask)
        for key in summary_1.keys():
            summary_1[key][hoi_id] = output_1[key]
print(
    "def full %.4f, def rare %.4f, def non-rare %.4f, ko full %.4f, ko rare %.4f, ko non-rare %.4f" % (
        np.mean(summary_1['ap']),    np.mean(summary_1['ap'][rare > 1]),    np.mean(summary_1['ap'][rare < 1]), 
        np.mean(summary_1['ap_ko']), np.mean(summary_1['ap_ko'][rare > 1]), np.mean(summary_1['ap_ko'][rare < 1])
    )
)