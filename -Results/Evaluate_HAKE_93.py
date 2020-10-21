import argparse
import pickle
from pasta_utils import Generate_PVP_detection, Generate_verb_detection
from tqdm import tqdm
import os
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--result-dir', default='', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Loading results...')
    keys, hbox, score, verb_score = [], [], [], []
    for each_img_res_file in tqdm(os.listdir(args.result_dir)):
        img_name = os.path.splitext(each_img_res_file)[0]
        each_img_res = pickle.load(open(os.path.join(args.result_dir, each_img_res_file),'rb'))
        for key_idx in range(len(each_img_res[0])):
            if each_img_res[0][key_idx][0] != 'collect':
                each_img_res[0][key_idx][1] = int(each_img_res[0][key_idx][1])
        keys.extend(each_img_res[0])
        hbox.extend(each_img_res[1])
        score.extend(each_img_res[2])
        verb_score.extend(each_img_res[3])
    freqs = np.load('freqs_verb.npy')
    head_dir_name, sub_dir_name = os.path.split(args.result_dir)
    results_dir = os.path.join(head_dir_name, sub_dir_name+'_results')
    os.makedirs(results_dir, exist_ok=True)
    gt_pasta_results = pickle.load(open('img_to_pairs_pvp.pkl','rb'))
    gt_verb_results = pickle.load(open('img_to_pairs_verb.pkl','rb'))

    print('*******************************************')
    print('* Calculating PaSta prediction results... *')
    print('*******************************************')

    Generate_PVP_detection(keys, hbox, score, gt_pasta_results, os.path.join(results_dir, 'map_rec_results_pvp.txt'))

    print('*******************************************')
    print('* Calculating verb prediction results...  *')
    print('*******************************************')
    
    Generate_verb_detection(keys, hbox, verb_score, gt_verb_results, os.path.join(results_dir, 'map_rec_results_verb.txt'), freqs)