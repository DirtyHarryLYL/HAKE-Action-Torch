#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#  Last Modified: Dec 9th, 2020             #
#############################################
import argparse
import os
import os.path as osp
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
import cv2
import imageio
import yaml
from easydict import EasyDict as edict
import copy
import pprint

from test_net import annos_to_cuda
from inference_tools.pose_inference import AlphaPose
from inference_tools.custom_multiprocessing import process_pool
from inference_tools.pasta_inference import pasta_model

from activity2vec.dataset.hake_dataset import im_read, rgba2rgb
from activity2vec.ult.config import get_cfg

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PART_COLOR_LIST = [GREEN, CYAN, YELLOW, ORANGE, PURPLE, RED]

BROWN = (128, 42, 42)
JACKIE_BLUE = (11, 23, 70)
YELLOW_BROWN = (240, 230, 140)
SOMECOLOR = (255, 127, 127)
STRAWBERRY = (135, 38, 87)
DARKGREEN = (48, 128, 20)
ID_COLOR_LIST = [DARKGREEN, BROWN, STRAWBERRY, JACKIE_BLUE, BLUE]
class Activity2Vec():
    def __init__(self, cfg):
        self.cfg = cfg
        self.pasta_name_list = np.array([x.strip() for x in open(cfg.DATA.PASTA_NAME_LIST).readlines()])
        self.verb_name_list = np.array([x.strip() for x in open(cfg.DATA.VERB_NAME_LIST).readlines()])
        self.excluded_verbs = cfg.DEMO.EXCLUDED_VERBS
        self.excluded_verb_names = np.delete(self.verb_name_list, self.excluded_verbs, axis=0)
        self.alphapose = AlphaPose(cfg.DEMO.YOLO_CFG, cfg.DEMO.YOLO_WEIGHT, cfg.DEMO.POSE_CFG, cfg.DEMO.POSE_WEIGHT)
        self.pasta_model = pasta_model(cfg)

    def inference(self, image_path, image=None):
        if image is None:
            ori_image = im_read(image_path)
            alpha_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        else:
            ori_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            alpha_image = image
        pose = self.alphapose.process(image_path, alpha_image)
        if pose is None:
            print('[AlphaPose] no pose result for {:s}'.format(image_path))
            vis = ori_image
            vis = self.visualize(vis, None, None, None, None, None)
            return ori_image, None, vis
        else:
            try:
                pasta_image, annos = self.pasta_model.preprocess(ori_image, pose['result'])
                annos_cpu = copy.deepcopy(annos)
                pasta_image = pasta_image.cuda(non_blocking=True)
                annos = annos_to_cuda(self.cfg, annos)
                
                f_pasta, p_pasta, p_verb = self.pasta_model.inference(pasta_image, annos)
                vis = ori_image
                if self.cfg.DEMO.DRAW_SKELETON:
                    vis = self.alphapose.vis(vis, pose)

                scores = annos_cpu['human_scores'][0].numpy()[:, 0]
                score_rank = np.argsort(scores, axis=0)[::-1]
                scores = scores[score_rank][:cfg.DEMO.MAX_HUMAN_NUM]
                bboxes = annos_cpu['human_bboxes'][0].numpy()[score_rank][:cfg.DEMO.MAX_HUMAN_NUM]#[np.where(scores>self.cfg.DEMO.HUMAN_SCORE_THRES)[0], :]
                keypoints = annos_cpu['keypoints'][0].numpy()[score_rank][:cfg.DEMO.MAX_HUMAN_NUM]#[np.where(scores>self.cfg.DEMO.HUMAN_SCORE_THRES)[0], :]
                p_pasta = p_pasta[score_rank][:cfg.DEMO.MAX_HUMAN_NUM]
                p_verb = p_verb[score_rank][:cfg.DEMO.MAX_HUMAN_NUM]
                vis = self.visualize(vis, bboxes, keypoints, scores, p_pasta, p_verb)#[np.where(scores>self.cfg.DEMO.HUMAN_SCORE_THRES)[0], :], p_verb[np.where(scores>self.cfg.DEMO.HUMAN_SCORE_THRES)[0], :])
                
                annos_cpu['f_pasta'] = f_pasta
                annos_cpu['p_pasta'] = p_pasta
                annos_cpu['p_verb'] = p_verb
            except:
                print('[AlphaPose] unsuccess for {:s}'.format(image_path))
                vis = ori_image
                vis = self.visualize(vis, None, None, None, None, None)
                return ori_image, None, vis
        return ori_image, annos_cpu, vis

    
    def visualize(self, image, human_bboxes, keypoints, human_scores, p_pasta, p_verb, topk=5):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_shape = list(image.shape)
        ones_shape = copy.deepcopy(im_shape)
        ones_shape[1] = 80
        image_ones = np.ones(ones_shape, dtype=image.dtype) * 0
        image = np.concatenate((image, image_ones), axis=1)
        pil_image = Image.fromarray(image).convert('RGBA')
        
        overlay = Image.new('RGBA', pil_image.size, WHITE+(0,))
        overlay_draw = ImageDraw.Draw(overlay)
        
        canvas = Image.new('RGBA', pil_image.size, WHITE+(0,))
        draw = ImageDraw.Draw(canvas)

        font = ImageFont.truetype(self.cfg.DEMO.FONT_PATH, self.cfg.DEMO.FONT_SIZE)
        font_id = ImageFont.truetype(self.cfg.DEMO.FONT_PATH, self.cfg.DEMO.FONT_SIZE)

        draw.text((im_shape[1]+1, 3), '────────', font=font, fill=CYAN+(255, ))
        if human_bboxes is not None:
            # human_areas = (human_bboxes[:, 2] - human_bboxes[:, 0]) * (human_bboxes[:, 3] - human_bboxes[:, 1])
            # max_human_area_idxs = np.argsort(human_areas)[::-1][:cfg.DEMO.MAX_HUMAN_NUM]

            human_count = 0
            extra_offset = 0
            extra_offset += self.cfg.DEMO.FONT_SIZE
            for idx in range(len(human_bboxes)):
                # if idx not in max_human_area_idxs:
                #     continue
                human_bbox = human_bboxes[idx]
                human_keypoints = keypoints[idx]
                pasta_scores = p_pasta[idx]
                verb_scores = p_verb[idx]    
                ori_verb_scores = copy.deepcopy(verb_scores)
                
                verb_scores = np.delete(verb_scores, self.excluded_verbs, axis=0)
                verb_top_idxs = np.argsort(verb_scores)[::-1]

                verb_draw_names = []
                for top_idx, verb_name in enumerate(self.excluded_verb_names[verb_top_idxs]):
                    verb_idx = verb_top_idxs[top_idx]
                    if verb_name not in verb_draw_names:
                        verb_draw_names.append(verb_name)
                    if len(verb_draw_names) == topk:
                        break

                pasta_draw_names = []
                for pasta_range_idx, pasta_range in enumerate(self.cfg.DATA.PASTA_RANGES):
                    pasta_sub_scores = pasta_scores[pasta_range[0]:pasta_range[1]]
                    pasta_names = self.pasta_name_list[pasta_range[0]:pasta_range[1]]
                    if self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'hip' and self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'leg' and self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'arm':
                        pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1]
                        for pasta_sub_idx in pasta_sub_idxs:
                            pasta_name = pasta_names[pasta_sub_idx]
                            if 'no_interaction' not in pasta_name:
                                pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                                break
                    elif self.cfg.DATA.PASTA_NAMES[pasta_range_idx] == 'arm':
                        pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1]
                        for pasta_sub_idx in pasta_sub_idxs:
                            pasta_name = pasta_names[pasta_sub_idx]
                            if 'no_interaction' not in pasta_name:
                                if 'be close to' not in pasta_name:
                                    pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                                else:
                                    pasta_draw_names.append('')
                                break
                    elif self.cfg.DATA.PASTA_NAMES[pasta_range_idx] == 'leg':
                            no_interaction_score = pasta_sub_scores[-1]
                            # print('### leg ###')
                            # print(pasta_sub_scores)
                        # if no_interaction_score > 0.8:
                        #     pasta_draw_names.append('')
                        # else:
                            pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1]
                            for pasta_sub_idx in pasta_sub_idxs:
                                pasta_name = pasta_names[pasta_sub_idx]
                                if 'no_interaction' not in pasta_name:
                                    if 'is close with' not in pasta_name:
                                        pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                                    else:
                                        pasta_draw_names.append('')
                                    break
                            # print('###########')
                        # if pasta_sub_scores[-1] > 0.95:
                        #     pasta_draw_names[-1] = ''
                    else:
                        sit_at_score = ori_verb_scores[86]
                        sit_on_score = ori_verb_scores[87]
                        no_interaction_score = pasta_sub_scores[-1]
                        # print('*** hip ***')
                        # print(idx, sit_at_score, sit_on_score, no_interaction_score, pasta_sub_scores)
                        if (sit_at_score > 0.05 or sit_on_score > 0.19 or no_interaction_score < 0.3) and 'sit' in verb_draw_names: # or no_interaction_score < 0.2 sit on score 0.19
                        # if 'sit' in verb_draw_names:
                            pasta_draw_names.append('sit')
                        else:
                            pasta_draw_names.append('')
                        # print('***********')
                        # pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1] 
                        # for pasta_sub_idx in pasta_sub_idxs:
                        #     pasta_name = pasta_names[pasta_sub_idx]
                        #     if 'no_interaction' not in pasta_name:
                        #         pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                        #         break
                    # if pasta_draw_names[-1] == 'no_interaction':
                    #     pasta_draw_names[-1] = ''
                # if pasta_draw_names[0] == 'stand' and pasta_draw_names[1] == 'jump' and 'jump' not in verb_draw_names[:3]:
                #     pasta_draw_names[1] = ''
                # if pasta_draw_names[1] == 'walk':
                #     pasta_draw_names[2] = ''
                # if pasta_draw_names[1] == 'is close with':
                #     pasta_draw_names[1] = ''
                # if pasta_draw_names[4] == 'be close to':
                #     pasta_draw_names[4] = ''
                # print(pasta_draw_names)

                # verb_offsets = len(verb_draw_names) * self.cfg.DEMO.FONT_SIZE
                # verb_max_length = max([len(verb_draw_name) for verb_draw_name in verb_draw_names])
                verb_offsets = 1 * self.cfg.DEMO.FONT_SIZE
                verb_max_length = 1.5
                # overlay_draw.rectangle([(int(human_bbox[0]), int(human_bbox[1])), (int(human_bbox[0]+verb_max_length * self.cfg.DEMO.FONT_SIZE / 1.7),int(human_bbox[1]+verb_offsets+3))], fill=WHITE+(128,))
                draw.rectangle([(int(human_bbox[0]), int(human_bbox[1])), (int(human_bbox[0]+verb_max_length * self.cfg.DEMO.FONT_SIZE / 1.7),int(human_bbox[1]+verb_offsets+3))], fill=ID_COLOR_LIST[human_count % 5]+(255,))
                
                # pasta_offsets = len(pasta_draw_names) * self.cfg.DEMO.FONT_SIZE
                # pasta_max_length = max([len(pasta_draw_name) for pasta_draw_name in pasta_draw_names])
                # overlay_draw.rectangle([(int(human_bbox[2]-pasta_max_length * self.cfg.DEMO.FONT_SIZE / 1.7),int(human_bbox[3]-pasta_offsets)), (int(human_bbox[2]), int(human_bbox[3]))], fill=WHITE+(128,))
                # pil_image = Image.alpha_composite(pil_image, overlay)

                # draw = ImageDraw.Draw(pil_image, 'RGBA')
                draw.rectangle([(int(human_bbox[0]), int(human_bbox[1])), (int(human_bbox[2]),int(human_bbox[3]))], fill=None, outline=BLUE+(255, ), width=2)

                draw.text((int(human_bbox[0])+3, int(human_bbox[1])+3), str(human_count), font=font_id, fill=WHITE+(255, ))
                # draw.text((im_shape[1]+1, 1+extra_offset), '--------', font=font, fill=YELLOW+(255, ))
                # extra_offset += self.cfg.DEMO.FONT_SIZE
                draw.text((im_shape[1]+1, 3+extra_offset), 'ID: '+str(human_count), font=font, fill=CYAN+(255, ))  # +' {:.3f}'.format(verb_scores[verb_idx])
                extra_offset += self.cfg.DEMO.FONT_SIZE
                for draw_name in verb_draw_names:
                    draw.text((im_shape[1]+1, 3+extra_offset), draw_name, font=font, fill=GREEN+(255, ))  # +' {:.3f}'.format(verb_scores[verb_idx])
                    extra_offset += self.cfg.DEMO.FONT_SIZE
                draw.text((im_shape[1]+1, 3+extra_offset), '────────', font=font, fill=CYAN+(255, ))
                extra_offset += self.cfg.DEMO.FONT_SIZE

                # extra_offset = 0
                # for draw_name in pasta_draw_names:
                #     draw.text((int(human_bbox[2]-pasta_max_length * self.cfg.DEMO.FONT_SIZE / 1.7)+3, int(human_bbox[3]-pasta_offsets)+extra_offset), draw_name, font=font, fill=BLUE+(255, ))  # +' {:.3f}'.format(verb_scores[verb_idx])
                #     extra_offset += self.cfg.DEMO.FONT_SIZE
                
                foot_and_hand = dict()
                no_foot_and_hand = dict()
                lfoot_width = None
                lhand_width = None
                rfoot_width = None
                rhand_width = None
                head_width = None
                for part_name, kps_idxs in self.cfg.DEMO.SKELETON_TO_PARTS.items():
                    if 'foot' in part_name or 'hand' in part_name:
                        foot_and_hand[part_name] = kps_idxs
                    else:
                        no_foot_and_hand[part_name] = kps_idxs
        
                for part_name, kps_idxs in no_foot_and_hand.items():
                    if part_name[1:] in self.cfg.DATA.PASTA_NAME_DICT:
                        part_idx = self.cfg.DATA.PASTA_NAME_DICT[part_name[1:]]

                    elif part_name in self.cfg.DATA.PASTA_NAME_DICT:
                        part_idx = self.cfg.DATA.PASTA_NAME_DICT[part_name]
                        
                    else:
                        raise NotImplementedError

                    kps_idxs = np.array(kps_idxs)
                    if 'foot' not in self.cfg.DATA.PASTA_NAMES[part_idx] and 'hand' not in self.cfg.DATA.PASTA_NAMES[part_idx] and 'head' not in self.cfg.DATA.PASTA_NAMES[part_idx]:
                        for pair_id in range(len(kps_idxs)-1):
                            start_idx = kps_idxs[pair_id]
                            end_idx = kps_idxs[pair_id+1]
                            start_keypoint = human_keypoints[start_idx][:2]
                            end_keypoint = human_keypoints[end_idx][:2]
                            score = (human_keypoints[start_idx][2] + human_keypoints[end_idx][2]) / 2
                            if score < 0.5:
                                continue
                            vec = end_keypoint - start_keypoint
                            norm_vec = np.array([vec[1], -vec[0]], dtype=vec.dtype) / 16
                            if 'hip' in self.cfg.DATA.PASTA_NAMES[part_idx]:
                                norm_vec *= 4
                            point1 = start_keypoint + norm_vec
                            point2 = start_keypoint - norm_vec
                            point3 = end_keypoint + norm_vec
                            point4 = end_keypoint - norm_vec
                            if cfg.DEMO.DRAW_RIGID:
                                draw.line(tuple(point1) + tuple(point3), fill=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                                draw.line(tuple(point2) + tuple(point4), fill=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                                draw.line(tuple(point1) + tuple(point2), fill=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                                draw.line(tuple(point3) + tuple(point4), fill=PART_COLOR_LIST[part_idx]+(255, ), width=2)

                            if pair_id == len(kps_idxs)-2:
                                draw_name = pasta_draw_names[part_idx]
                                
                                if 'hip' in self.cfg.DATA.PASTA_NAMES[part_idx]:
                                    overlay_draw.rectangle([(int((start_keypoint[0]+end_keypoint[0])/2-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int((start_keypoint[1]+end_keypoint[1])/2-self.cfg.DEMO.FONT_SIZE/2)), (int((start_keypoint[0]+end_keypoint[0])/2+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int((start_keypoint[1]+end_keypoint[1])/2+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                                    draw.text((((start_keypoint[0]+end_keypoint[0])/2-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int((start_keypoint[1]+end_keypoint[1])/2-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))
                                else:
                                    overlay_draw.rectangle([(int(start_keypoint[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(start_keypoint[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(start_keypoint[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(start_keypoint[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                                    draw.text((int(start_keypoint[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(start_keypoint[1])-self.cfg.DEMO.FONT_SIZE/2), draw_name, font=font, fill=BLUE+(255, ))
                                if part_name == 'lleg':
                                    lfoot_width = 2 * (np.sqrt(np.sum(norm_vec ** 2)))
                                if part_name == 'rleg':
                                    rfoot_width = 2 * (np.sqrt(np.sum(norm_vec ** 2)))
                                if part_name == 'larm':
                                    lhand_width = 2 * (np.sqrt(np.sum(norm_vec ** 2)))
                                if part_name == 'rarm':
                                    rhand_width = 2 * (np.sqrt(np.sum(norm_vec ** 2)))
                                
                    elif 'head' in self.cfg.DATA.PASTA_NAMES[part_idx]:
                        
                        head_keypoints = human_keypoints[kps_idxs]
                        x1, x2 = head_keypoints.min(axis=0)[0], head_keypoints.max(axis=0)[0]
                        head_width = x2-x1
                        head_centre = head_keypoints[0][:2]
                        head_score = head_keypoints[0][2]
                        if head_score < 0.5:
                            continue
                        draw_name = pasta_draw_names[part_idx]
                        draw.rectangle([(int(head_centre[0]-head_width/2), int(head_centre[1]-head_width/2)), (int(head_centre[0]+head_width/2), int(head_centre[1]+head_width/2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                        overlay_draw.rectangle([(int(head_centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(head_centre[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(head_centre[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(head_centre[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                        draw.text((int(head_centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(head_centre[1]-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))
                        
                for part_name, kps_idxs in foot_and_hand.items():
                    if part_name[1:] in self.cfg.DATA.PASTA_NAME_DICT:
                        part_idx = self.cfg.DATA.PASTA_NAME_DICT[part_name[1:]]

                    elif part_name in self.cfg.DATA.PASTA_NAME_DICT:
                        part_idx = self.cfg.DATA.PASTA_NAME_DICT[part_name]
                        
                    else:
                        raise NotImplementedError
                    draw_name = pasta_draw_names[part_idx]
                    kps_idx = kps_idxs[0]
                    centre = human_keypoints[kps_idx][:2]
                    score = human_keypoints[kps_idx][2]
                    if score < 0.5:
                        continue
                    if 'rfoot' in part_name and rfoot_width is not None:
                        draw.rectangle([(int(centre[0]-rfoot_width*2), int(centre[1]-rfoot_width*2)), (int(centre[0]+rfoot_width*2), int(centre[1]+rfoot_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'lfoot' in part_name and lfoot_width is not None:
                        draw.rectangle([(int(centre[0]-lfoot_width*2), int(centre[1]-lfoot_width*2)), (int(centre[0]+lfoot_width*2), int(centre[1]+lfoot_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'rhand' in part_name and rhand_width is not None:
                        draw.rectangle([(int(centre[0]-rhand_width*2), int(centre[1]-rhand_width*2)), (int(centre[0]+rhand_width*2), int(centre[1]+rhand_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'lhand' in part_name and lhand_width is not None:
                        draw.rectangle([(int(centre[0]-lhand_width*2), int(centre[1]-lhand_width*2)), (int(centre[0]+lhand_width*2), int(centre[1]+lhand_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    draw.text((int(centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(centre[1]-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))
                    overlay_draw.rectangle([(int(centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(centre[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(centre[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(centre[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                
                lshoulder_centre = human_keypoints[5][:2]
                lshoulder_score  = human_keypoints[5][2]
                rshoulder_centre = human_keypoints[6][:2]
                rshoulder_score  = human_keypoints[6][2]
                neck_centre = (lshoulder_centre + rshoulder_centre) / 2
                hip_centre  = (human_keypoints[11][:2] + human_keypoints[12][:2]) / 2

                start_keypoint = lshoulder_centre
                end_keypoint = rshoulder_centre
                vec = end_keypoint - start_keypoint
                norm_vec = np.array([vec[1], -vec[0]], dtype=vec.dtype) / 16
                point1 = start_keypoint + norm_vec
                point2 = start_keypoint - norm_vec
                point3 = end_keypoint + norm_vec
                point4 = end_keypoint - norm_vec
                if cfg.DEMO.DRAW_RIGID:
                    draw.line(tuple(point1) + tuple(point3), fill=PURPLE+(255, ), width=2)
                    draw.line(tuple(point2) + tuple(point4), fill=PURPLE+(255, ), width=2)
                    draw.line(tuple(point1) + tuple(point2), fill=PURPLE+(255, ), width=2)
                    draw.line(tuple(point3) + tuple(point4), fill=PURPLE+(255, ), width=2)

                start_keypoint = neck_centre
                end_keypoint = hip_centre
                vec = end_keypoint - start_keypoint
                norm_vec = np.array([vec[1], -vec[0]], dtype=vec.dtype) / 24
                point1 = start_keypoint + norm_vec
                point2 = start_keypoint - norm_vec
                point3 = end_keypoint + norm_vec
                point4 = end_keypoint - norm_vec
                if cfg.DEMO.DRAW_RIGID:
                    draw.line(tuple(point1) + tuple(point3), fill=SOMECOLOR+(255, ), width=2)
                    draw.line(tuple(point2) + tuple(point4), fill=SOMECOLOR+(255, ), width=2)
                    draw.line(tuple(point1) + tuple(point2), fill=SOMECOLOR+(255, ), width=2)
                    draw.line(tuple(point3) + tuple(point4), fill=SOMECOLOR+(255, ), width=2)

                human_count += 1

        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = Image.alpha_composite(pil_image, canvas)
        pil_image = pil_image.convert('RGB')
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return cv2_image

def parse_args(cfg):

    parser = argparse.ArgumentParser(description='Activity2Vec Demo')

    parser.add_argument('--input', type=str, required=True, 
                        help='input path/directory')
    parser.add_argument('--output', type=str, required=True, 
                        help='output path/directory')

    parser.add_argument('--mode', type=str, choices=['image', 'video'], default='image',
                        help='choose the type of input')
    parser.add_argument('--gpus', type=str, default='0', 
                        help='select cuda devices')
    parser.add_argument('--yolo-cfg', type=str, default=cfg.DEMO.YOLO_CFG, 
                        help='the path of yolo configuration file')
    parser.add_argument('--yolo-weight', type=str, default=cfg.DEMO.YOLO_WEIGHT,
                        help='the path of yolo weight')
    parser.add_argument('--pose-cfg', type=str, default=cfg.DEMO.POSE_CFG,
                        help='the path of alphapose configuration file')
    parser.add_argument('--pose-weight', type=str, default=cfg.DEMO.POSE_WEIGHT,
                        help='the path of alphapose weight')
    parser.add_argument('--a2v-cfg', type=str, default=cfg.DEMO.A2V_CFG,
                        help='the path of a2v configuration file')
    parser.add_argument('--a2v-weight', type=str, default=cfg.DEMO.A2V_WEIGHT,
                        help='the path of a2v weight')
    parser.add_argument('--draw-skeleton', action='store_true', 
                        help='choose whether to draw skeleton')
    parser.add_argument('--human-score-thres', type=float, default=cfg.DEMO.HUMAN_SCORE_THRES,
                        help='threshold of human detection score')
    args = parser.parse_args()
    return args

def update_cfg(cfg, args):

    def merge_a_into_b(a, b):
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                assert isinstance(b[k], dict)
                merge_a_into_b(v, b[k])
            else:
                b[k] = v
    
    cfg.DEMO.A2V_CFG = args.a2v_cfg
    loaded_cfg = yaml.load(open(cfg.DEMO.A2V_CFG, 'r'))
    merge_a_into_b(loaded_cfg, cfg)

    cfg.DEMO.YOLO_CFG = args.yolo_cfg
    cfg.DEMO.YOLO_WEIGHT = args.yolo_weight
    cfg.DEMO.POSE_CFG = args.pose_cfg
    cfg.DEMO.POSE_WEIGHT = args.pose_weight
    cfg.DEMO.A2V_WEIGHT = args.a2v_weight
    cfg.DEMO.DRAW_SKELETON = args.draw_skeleton
    cfg.DEMO.HUMAN_SCORE_THRES = args.human_score_thres

    return cfg

def check_img(filename):
    return True if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')) else False

def read_video(args):
    video_path = args.input
    clip = VideoFileClip(video_path)
    frames = []
    print('Reading frames ...')
    for frame in tqdm(clip.iter_frames()):
        frames.append(frame)
    return frames

def read_input(args):
    if args.mode == 'image':
        filepath = args.input
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                print('[Input] Image directory detected.')
                imgpaths = []
                for root, dirs, files in os.walk(filepath):
                    if len(dirs) == 0:
                        for file in files:
                            if check_img(file):
                                imagepath = os.path.join(root, file)
                                imgpaths.append(imagepath)
            elif check_img(filepath):
                print('[Input] Image file detected.')
                imgpaths = [filepath]
            
            elif filepath.lower().endswith('.txt'):
                print('[Input] Image list detected.')
                imgpaths = open(filepath, 'r').readlines()
                imgpaths = [x.strip() for x in imgpaths]
            
            else:
                print('[Input] Not supported file format {:s}'.format(os.path.splitext(filepath)[-1]))
                raise NotImplementedError
            return imgpaths
        else:
            print('[Input] path does not exist! Empty image list will be returned.')
            return []
    else:
        return read_video(args)
        
if __name__ == '__main__':
    cfg = get_cfg()
    args = parse_args(cfg)
    cfg = update_cfg(cfg, args)
    print('cfg:\n'+pprint.pformat(cfg, indent=2))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    a2v = Activity2Vec(cfg)
    image_list = read_input(args)

    if args.mode == 'image':
        for image_path in image_list:
            ori_image, _, vis = a2v.inference(image_path)
            if vis is not None:
                cv2.imshow('Activity2Vec', vis)
            else:
                cv2.imshow('Activity2Vec', ori_image)
            cv2.waitKey(0)
    else:
        vises = []
        for idx, image in enumerate(tqdm(image_list[:120])):
            ori_image, _, vis = a2v.inference('%d.jpg' % idx, image)
            vises.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        new_clip = ImageSequenceClip(vises, fps=25)
        if os.path.isdir(args.output):
            new_clip.write_videofile(os.path.join(args.output, os.path.basename(args.input)))
        else:
            new_clip.write_videofile(args.output)
            # if idx % 30 == 0:
            #     ori_image, _, vis = a2v.inference('%d.jpg' % idx, image)
            #     if vis is not None:
            #         cv2.imshow('Activity2Vec', vis)
            #     else:
            #         cv2.imshow('Activity2Vec', ori_image)
            #     cv2.waitKey(0)