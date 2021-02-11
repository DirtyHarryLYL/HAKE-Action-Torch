##################################################################################
#  Author: Hongwei Fan                                                           #
#  E-mail: hwnorm@outlook.com                                                    #
#  Homepage: https://github.com/hwfan                                            #
#  Based on PaStaNet in CVPR'20                                                  #
#  TF version:                                                                   #
#  https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action  #
##################################################################################
import pickle
from tqdm import tqdm
import os
import numpy as np
import h5py
import os.path as osp
def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union

def calc_ap(recall, predict):
    ap = 0.0
    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mask = recall >= i
        if np.sum(mask) <= 0:
            continue
        ap += np.max(predict[mask])
    ap /= 11.0
    return ap

def calc_map(gt, pred_keys, pred_bboxes, pred_scores, verb_idx, recall_total, thres=0.3):
    if verb_idx not in gt or verb_idx not in recall_total:
        return np.nan
    sorted_idxs = np.argsort(pred_scores[:, verb_idx])[::-1]
    sorted_keys = pred_keys[sorted_idxs]
    sorted_bboxes = pred_bboxes[sorted_idxs, :]
    sorted_scores = pred_scores[sorted_idxs, verb_idx]

    predict_array = np.zeros(sorted_scores.shape[0], dtype=int)
    for prediction_idx in range(len(sorted_scores)):
        
        key = sorted_keys[prediction_idx]
        if key not in gt[verb_idx]:
            predict_array[prediction_idx] = 0
            continue

        gt_bboxes = gt[verb_idx][key][0].astype(np.float32)
        pred_bboxes = sorted_bboxes[prediction_idx:prediction_idx+1]
        iou_matrix = calc_iou(pred_bboxes, gt_bboxes)
        iou_array = iou_matrix[0]
        hit_array = iou_array > thres

        max_idx = -1
        max_iou = 0
        for hit_idx in np.where(hit_array == 1)[0]:
            if iou_array[hit_idx] > max_iou and gt[verb_idx][key][1][hit_idx] == 0:
                max_iou = iou_array[hit_idx]
                max_idx = hit_idx
        if max_idx != -1:
            predict_array[prediction_idx] = 1
            gt[verb_idx][key][1][max_idx] = 1    
        else:
            predict_array[prediction_idx] = 0
    predict = np.cumsum(predict_array) / (np.arange(len(predict_array)) + 1)
    recall = np.cumsum(predict_array) / recall_total[verb_idx]
    ap = calc_ap(recall, predict)
    return ap

def load_list(list_txt):
    lines = []
    for line in open(list_txt, 'r'):
        lines.append(line.strip())
    return lines

def benchmark(results_dir, cfg, logger):
    gt_pasta_data = pickle.load(open(cfg.DATA.TEST_GT_PASTA_PATH, 'rb'))
    gt_verb_data = pickle.load(open(cfg.DATA.TEST_GT_VERB_PATH, 'rb'))
    pasta_name_list = load_list(cfg.DATA.PASTA_NAME_LIST)
    verb_name_list = load_list(cfg.DATA.VERB_NAME_LIST)

    upper_names = [name.upper() for name in cfg.DATA.PASTA_NAMES]
    pasta_num_list = [0] + [cfg.DATA.NUM_PASTAS[upper_name] for upper_name in upper_names]
    pasta_num_list = np.array(pasta_num_list)
    pasta_num_list = np.cumsum(pasta_num_list)
    pasta_range_starts = list(pasta_num_list[:-1])
    pasta_range_ends   = list(pasta_num_list[1:])
    pasta_ranges       = list(zip(pasta_range_starts, pasta_range_ends))

    pasta_recall_total = dict()
    verb_recall_total = dict()

    for verb_idx in gt_pasta_data:
        for key in gt_pasta_data[verb_idx]:
            if verb_idx not in pasta_recall_total:
                pasta_recall_total[verb_idx] = gt_pasta_data[verb_idx][key].shape[0]
            else:
                pasta_recall_total[verb_idx] += gt_pasta_data[verb_idx][key].shape[0]
            gt_pasta_data[verb_idx][key] = [gt_pasta_data[verb_idx][key], np.zeros(gt_pasta_data[verb_idx][key].shape[0], dtype=int)]

    for verb_idx in gt_verb_data:
        for key in gt_verb_data[verb_idx]:
            if verb_idx not in verb_recall_total:
                verb_recall_total[verb_idx] = gt_verb_data[verb_idx][key].shape[0]
            else:
                verb_recall_total[verb_idx] += gt_verb_data[verb_idx][key].shape[0]
            gt_verb_data[verb_idx][key] = [gt_verb_data[verb_idx][key], np.zeros(gt_verb_data[verb_idx][key].shape[0], dtype=int)]

    key_list = []
    human_bboxes_list = []
    pasta_score_list = []
    verb_score_list = []
    logger.info('==> Collecting results ...')
    for root, dirs, files in os.walk(results_dir):
        if len(dirs) == 0:
            for result_file_path in tqdm(files, ncols=40):
                f = h5py.File(os.path.join(root, result_file_path),'r')
                key = root.split('/')[-1] + '/' + os.path.splitext(result_file_path)[0]

                human_bboxes = np.array(f['human_bboxes'])[0]
                pasta_score = np.array(f['pasta_score'])
                verb_score = np.array(f['verb_score'])
                f.close()

                this_key_list = [key for _ in range(len(human_bboxes))]
                key_list.extend(this_key_list)

                if human_bboxes_list is None:
                    human_bboxes_list = [human_bboxes]
                else:
                    human_bboxes_list.append(human_bboxes)
                
                if pasta_score_list is None:
                    pasta_score_list = [pasta_score]
                else:
                    pasta_score_list.append(pasta_score)
                
                if verb_score_list is None:
                    verb_score_list = [verb_score]
                else:
                    verb_score_list.append(verb_score)
    key_list = np.array(key_list)
    human_bboxes_list = np.concatenate(human_bboxes_list, axis=0)
    pasta_score_list = np.concatenate(pasta_score_list, axis=0)
    verb_score_list = np.concatenate(verb_score_list, axis=0)
    
    logger.info('==> Evaluating ...')
    map_list = []
    for pasta in tqdm(range(sum(list(cfg.DATA.NUM_PASTAS.values()))), ncols=40):
        map_res = calc_map(gt_pasta_data, key_list, human_bboxes_list, pasta_score_list, pasta, pasta_recall_total)
        map_list.append(map_res)
    map_list = np.array(map_list)
    if cfg.BENCHMARK.SHOW_ACTION_RES:
        logger.info('detailed pasta results:')
        for pasta_idx, pasta_map in enumerate(map_list):
            pasta_name = pasta_name_list[pasta_idx]
            logger.info('%s: %2.2f' % (pasta_name, pasta_map*100))

    logger.info('mAP Results:')
    map_w_no_interaction_list = []
    map_wo_no_interaction_list = []
    for part_idx in range(len(cfg.DATA.PASTA_NAMES)):
        map_w_no_interaction = np.nanmean(map_list[pasta_ranges[part_idx][0]:pasta_ranges[part_idx][1]]) * 100
        map_wo_no_interaction = np.nanmean(map_list[pasta_ranges[part_idx][0]:pasta_ranges[part_idx][1]-1]) * 100
        map_w_no_interaction_list.append(map_w_no_interaction)
        map_wo_no_interaction_list.append(map_wo_no_interaction)
        logger.info('%s: %2.2f, %2.2f' % (cfg.DATA.PASTA_NAMES[part_idx], map_w_no_interaction, map_wo_no_interaction))
    
    total_w_no_interaction = np.nanmean(np.array(map_w_no_interaction_list))
    total_wo_no_interaction = np.nanmean(np.array(map_wo_no_interaction_list))
    logger.info('pasta: %2.2f, %2.2f' % (total_w_no_interaction, total_wo_no_interaction))
    verb_map_list = []
    for verb in tqdm(range(cfg.DATA.NUM_VERBS), ncols=40):
        map_res = calc_map(gt_verb_data, key_list, human_bboxes_list, verb_score_list, verb, verb_recall_total)
        verb_map_list.append(map_res)

    verb_mean_ap = np.nanmean(np.array(verb_map_list)) * 100
    verb_mean_ap_wo_no_interaction = np.nanmean(np.array(verb_map_list[:57] + verb_map_list[58:])) * 100
    logger.info('verb-157: %2.2f, %2.2f' % (verb_mean_ap, verb_mean_ap_wo_no_interaction))

    verb_mean_ap_117 = np.nanmean(np.array(verb_map_list[:117])) * 100
    verb_mean_ap_117_wo_no_interaction = np.nanmean(np.array(verb_map_list[:57] + verb_map_list[58:117])) * 100
    logger.info('verb-117: %2.2f, %2.2f' % (verb_mean_ap_117, verb_mean_ap_117_wo_no_interaction))
    
    import ipdb; ipdb.set_trace()
    if cfg.BENCHMARK.SHOW_ACTION_RES:
        logger.info('detailed verb results:')
        for verb_idx, verb_map in enumerate(verb_map_list):
            verb_name = verb_name_list[verb_idx]
            logger.info('%s: %2.2f' % (verb_name, verb_map*100))

    return total_w_no_interaction, verb_mean_ap, map_w_no_interaction_list, map_wo_no_interaction_list
