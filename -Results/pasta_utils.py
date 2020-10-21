import numpy as np
import os

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

def calc_hit(hbox, gt_hbox):
    iou_out = iou(hbox, gt_hbox)
    return iou_out

def calc_PVP93(keys, hbox, score, gt, id):
    if id not in gt:
        return np.nan, np.nan#0, 0
    gt_label = gt[id]
    used = {}
    sum = 0
    hit, recog = [], []
    for key in gt_label.keys():
        used[key] = set()
        sum += len(gt_label[key])
    if sum == 0:
        return np.nan, np.nan
        
    idx = np.argsort(score, axis=0)[::-1]
    for i_idx in range(len(idx)):
        pair_id = idx[i_idx]
        if keys[pair_id][1] not in gt_label: # keys: image_id
            hit.append(0)
            recog.append(0)
        else:
            maximum = 0.0
            hit_idx = -1
            for i in range(len(gt_label[keys[pair_id][1]])):
                item = gt_label[keys[pair_id][1]][i]
                tmp = calc_hit(hbox[pair_id][0][:4], item[0])
                if tmp > 0.3:
                    if tmp > maximum:
                        maximum = tmp
                        hit_idx = i
            if hit_idx == -1:
                hit.append(0)
                recog.append(0)
            elif hit_idx in used[keys[pair_id][1]]:
                hit.append(1)
                recog.append(0)
            else:
                used[keys[pair_id][1]].add(hit_idx)
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
    rec = np.max(rec) if len(rec) > 0 else 0.0
    return ap, rec

def Generate_verb_detection(keys, hbox, score, gt, HICO_dir, freqs):
    map  = np.zeros(157, dtype='float')
    rec = np.zeros(157, dtype='float')
    verb_list = []
    for line in open('verb_list.txt','r'):
        verb = line.strip()
        verb_list.append(verb)
    with open(HICO_dir, 'w') as f:
        for i in range(157):
            if int(freqs[i]) == 0:
                map[i], rec[i] = np.nan, np.nan
                continue
            in_score = [each_score[0][i] for each_score in score]
            map[i], rec[i] = calc_PVP93(keys, hbox, in_score, gt, i)
            print(verb_list[i], "AP:", map[i], "REC:", rec[i])
            f.write('Name:%s, AP:%.4f, REC:%.4f\n' % (verb_list[i], map[i], rec[i]))
        
        f.write('All,  AP:%.4f, REC:%.4f\n' % (
                    float(np.nanmean(map)), 
                    float(np.nanmean(rec))
                ))
    
def Generate_PVP_detection(keys, hbox, score, gt, HICO_dir):
    map  = np.zeros(93, dtype='float')
    rec = np.zeros(93, dtype='float')
    pvp_list = []
    for line in open('Part_State_93.txt','r'):
        pvp = line.strip()
        pvp_list.append(pvp)
    for i in range(93):
        in_score = np.array([each_score[0][i] for each_score in score])
        map[i], rec[i] = calc_PVP93(keys, hbox, in_score, gt, i)
        print(pvp_list[i], "AP:", map[i], "REC:", rec[i])
    
    with open(HICO_dir, 'w') as f:
        f.write('PVP93 Evaluation\n')
        for i in range(93):
            f.write('Name:%s, AP:%.4f, REC:%.4f\n' % (pvp_list[i], float(map[i]), float(rec[i])))
        f.write(
            'Foot, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[:16])), 
                float(np.nanmean(map[:15])), 
                float(np.nanmean(rec[:16])), 
                float(np.nanmean(rec[:15]))
            )
        )
        f.write(
            'Leg,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[16:31])), 
                float(np.nanmean(map[16:30])), 
                float(np.nanmean(rec[16:31])), 
                float(np.nanmean(rec[16:30]))
            )
        )
        f.write(
            'Hip,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[31:37])), 
                float(np.nanmean(map[31:36])), 
                float(np.nanmean(rec[31:37])), 
                float(np.nanmean(rec[31:36]))
            )
        )
        f.write(
            'Hand, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[37:71])), 
                float(np.nanmean(map[37:70])), 
                float(np.nanmean(rec[37:71])), 
                float(np.nanmean(rec[37:70]))
            )
        )
        f.write(
            'Arm,  AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[71:79])), 
                float(np.nanmean(map[71:78])), 
                float(np.nanmean(rec[71:79])), 
                float(np.nanmean(rec[71:78]))
            )
        )
        f.write(
            'Head, AP:%.4f(%.4f), REC:%.4f(%.4f)\n' % (
                float(np.nanmean(map[79:])), 
                float(np.nanmean(map[79:92])), 
                float(np.nanmean(rec[79:])), 
                float(np.nanmean(rec[79:92]))
            )
        )
        
        f.write(
            'All,  AP:%.4f, REC:%.4f\n' % (
                float(np.nanmean(map)), 
                float(np.nanmean(rec))
            )
        )
