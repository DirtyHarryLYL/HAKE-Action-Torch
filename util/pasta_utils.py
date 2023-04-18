import numpy as np
import torch
import pickle
import os

P_num = {
    'P0': 12,
    'P1': 10,
    'P2': 5,
    'P3': 31,
    'P4': 5,
    'P5': 13,
    'labels_r': 600
}

pvp_weight = {
    'P0': torch.from_numpy(np.array([ 43.2812,  49.5520,  69.6451,  77.5116,  88.3417,  95.5558,  76.5148,  93.8608,  103.3516,  60.6352,  92.0465,  25.2341,  ])).float(),
    'P1': torch.from_numpy(np.array([ 69.5596,  77.4802,  88.1881,  95.5558,  60.6139,  65.1348,  48.4978,  103.3516,  92.0915,  24.0444,  ])).float(),
    'P2': torch.from_numpy(np.array([ 36.3299,  56.6936,  46.2726,  65.2572,  27.7231,  ])).float(),
    'P3': torch.from_numpy(np.array([ 32.5367,  72.1672,  63.7646,  61.0690,  61.1483,  90.1340,  84.7626,  83.8529,  82.5052,  97.7554,  94.3004,  104.3744,  104.0713,  66.0379,  92.7910,  90.7056,  75.8540,  84.6549,  76.3548,  67.6484,  67.5938,  76.3361,  75.2008,  68.9715,  77.6701,  88.5925,  80.4797,  75.6431,  83.4084,  96.2822,  29.3323,  ])).float(),
    'P4': torch.from_numpy(np.array([ 61.9114,  65.4154,  65.6752,  70.1946,  23.5435,  ])).float(),
    'P5': torch.from_numpy(np.array([ 65.3873,  46.7805,  85.3648,  106.6058,  69.8477,  77.9509,  69.3396,  85.8354,  85.9080,  82.4364,  92.2733,  68.3823,  24.3365,  ])).float(),
    'labels_r': torch.from_numpy(HO_weight).float(),

}
names = [
    'stands on','treads step on','walks with','walks to','runs with','runs to',
    'dribbles','kicks','jumps down','jumps with','walks away','no interaction',
    'walks with','walks to','runs with','runs to','jumps with','closes with',
    'straddles','jumps down','walks away','no interaction',
    'sits on','sits in','sits beside','is close with','no interaction',
    'holds','carries','reaches for','touches','puts on','twists','wears','throws',
    'throws out','writes on','points with','points to','uses sth. point to','presses',
    'squeezes','scratches','pinches','gestures to','pushes','pulls','pulls with sth.',
    'washes','washes with sth.','holds in both hands','lifts','raises','feeds',
    'cuts with sth.','catches with sth.','pours into','no interaction',
    'carries','close to','hugs','swings','no interaction',
    'eats','inspects','talks with','talks to','closes with','kisses',
    'puts sth. over','licks','blows','drinks with','smells','wears','no interaction',
]
ignore = [11, 21, 26, 57, 62, 75]
no_in = []
for i in range(12):
    no_in.append(11)
for i in range(12, 22):
    no_in.append(21)
for i in range(22, 27):
    no_in.append(26)
for i in range(27, 58):
    no_in.append(57)
for i in range(58, 63):
    no_in.append(62)
for i in range(63, 76):
    no_in.append(75)


def iou(bb1, bb2):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    
    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0
    
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)

def calc_hit(hbox, obox, hobox):
    hiou = iou(hbox, hobox[:4])
    oiou = iou(obox, hobox[4:])
    return max(hiou, oiou)
    
def calc_PVP76(keyss, hboxes, oboxes, scores, obj, id):
    gt_label = pickle.load(open('PVP_mat/PVP_%d.pkl' % id, 'rb'))
    used = {}
    sum = 0
    hit, recog = [], []
    for key in gt_label.keys():
        used[key] = set()
        sum += len(gt_label[key])
    if sum == 0:
        return np.nan, np.nan
        
    obj_mask = pickle.load(open('obj_mask.pkl', 'rb'))
    obj_bool = obj_mask[id, obj]
    select   = np.where(obj_bool > 0)[0]
    if len(select) == 0:
        return 0, 0
    keys  = keyss[select]
    hbox  = hboxes[select, :]
    obox  = oboxes[select, :]
    score = scores[select]

    idx = np.argsort(score, axis=0)[::-1]
    for i_idx in range(len(idx)):
        pair_id = idx[i_idx]
        if keys[pair_id] not in gt_label:
            hit.append(0)
            recog.append(0)
        else:
            maximum = 0.0
            hit_idx = -1
            for i in range(len(gt_label[keys[pair_id]])):
                item = gt_label[keys[pair_id]][i]
                tmp = calc_hit(hbox[pair_id], obox[pair_id], item[0, :])
                if tmp > 0.3:
                    if tmp > maximum:
                        maximum = tmp
                        hit_idx = i
            if hit_idx == -1:
                hit.append(0)
                recog.append(0)
            elif hit_idx in used[keys[pair_id]]:
                hit.append(1)
                recog.append(0)
            else:
                used[keys[pair_id]].add(hit_idx)
                hit.append(1)
                recog.append(1)

    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(np.array(hit))
    recog  = np.cumsum(np.array(recog))
    rec    = recog / sum
    prec   = hit / bottom
    ap     = 0.0
    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mask = rec >= i
        if np.sum(mask) <= 0:
            continue
        ap += np.max(prec[mask])
    ap    /= 11.0
    return ap, np.max(rec)

def get_map(keys, bboxes, scores, obj, part):
    mapping = {
        'P0': (0, 12), 
        'P1': (12, 22), 
        'P2': (22, 27), 
        'P3': (27, 58), 
        'P4': (58, 63), 
        'P5': (63, 76), 
    }
    x, y = mapping[part]
    map = np.zeros(y - x)
    for id in range(x, y):
        map[id - x], _ = calc_PVP76(keys, bboxes[:, :4], bboxes[:, 4:], scores[:, id - x], obj, id)
    return map

def Generate_PVP_detection(output_file, HICO_dir):

    print('the output file is', output_file)
    
    with h5py.File('Test_1100000_HICO_DET_GRWM_HRS_0.8_0.3.h5', 'r') as f:
        keys = f['key'][...]
        hbox = f['H'][...]
        obox = f['O'][...]
        score = f['PVP'][:, :]
        SH   = f['SH'][:]
        SO   = f['SO'][:]
        obj  = f['obj'][:] - 1
        score = score * SH[:, None] * SO[:, None]
    map  = np.zeros(76, dtype='float')
    rec = np.zeros(76, dtype='float')

    for i in range(76):
        map[i], rec[i] = calc_PVP76(keys, hbox, obox, score, obj, i)
        print(map[i], rec[i])
    
    with open(HICO_dir, 'w') as f:
        f.write('PVP76 Evaluation\n')
        for i in range(76):
            f.write('Name:%22s, AP:%.4f, REC:%.4f\n' % (names[i], float(map[i]), float(rec[i])))
        f.write(
            'Foot, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[:12])), 
                float(np.nanmean(map[:11])), 
                float(np.nanmean(rec[:12])), 
                float(np.nanmean(rec[:11]))
            )
        )
        f.write(
            'Leg,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[12:22])), 
                float(np.nanmean(map[12:21])), 
                float(np.nanmean(rec[12:22])), 
                float(np.nanmean(rec[12:21]))
            )
        )
        f.write(
            'Hip,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[22:27])), 
                float(np.nanmean(map[22:26])), 
                float(np.nanmean(rec[22:27])), 
                float(np.nanmean(rec[22:26]))
            )
        )
        f.write(
            'Hand, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[27:58])), 
                float(np.nanmean(map[27:57])), 
                float(np.nanmean(rec[27:58])), 
                float(np.nanmean(rec[27:57]))
            )
        )
        f.write(
            'Arm,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[58:63])), 
                float(np.nanmean(map[58:62])), 
                float(np.nanmean(rec[58:63])), 
                float(np.nanmean(rec[58:62]))
            )
        )
        f.write(
            'Head, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[63:])), 
                float(np.nanmean(map[63:75])), 
                float(np.nanmean(rec[63:])), 
                float(np.nanmean(rec[63:75]))
            )
        )
        
        f.write(
            'All,  AP:%.4f, REC:%.4f\n' % (
                float(np.nanmean(map)), 
                float(np.nanmean(rec))
            )
        )