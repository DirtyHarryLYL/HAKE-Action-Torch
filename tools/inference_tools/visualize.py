import cv2
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

class vis_tool():
    def __init__(self, cfg, mode):
        self.mode = mode
        self.cfg = cfg
        self.pasta_name_list = np.array([x.strip() for x in open(cfg.DATA.PASTA_NAME_LIST).readlines()])
        self.verb_name_list = np.array([x.strip() for x in open(cfg.DATA.VERB_NAME_LIST).readlines()])
        self.excluded_verbs = cfg.DEMO.EXCLUDED_VERBS
        self.excluded_verb_names = np.delete(self.verb_name_list, self.excluded_verbs, axis=0)
        self.skeleton_to_parts = {'lfoot': [16], 'rfoot': [15], 'lleg': [11, 13, 15], 'rleg': [12, 14, 16], 'hip': [12, 11], 'rhand': [9], 'lhand': [10], 'rarm': [5, 7, 9], 'larm': [6, 8, 10], 'head': [0, 1, 2, 3, 4]}
        self.pasta_name_dict = {part_name:part_idx for part_idx, part_name in enumerate(cfg.DATA.PASTA_NAMES)}
        upper_names = [name.upper() for name in cfg.DATA.PASTA_NAMES]
        pasta_num_list = [0] + [cfg.DATA.NUM_PASTAS[upper_name] for upper_name in upper_names]
        pasta_num_list = np.array(pasta_num_list)
        pasta_num_list = np.cumsum(pasta_num_list)
        pasta_range_starts = list(pasta_num_list[:-1])
        pasta_range_ends   = list(pasta_num_list[1:])
        self.pasta_ranges       = list(zip(pasta_range_starts, pasta_range_ends))

    def rotated_rectangle(self, drawer, start_keypoint, end_keypoint, color, norm_vec_scale=1.0):
        vec = end_keypoint - start_keypoint
        norm_vec = np.array([vec[1], -vec[0]], dtype=vec.dtype) / 16
        norm_vec *= norm_vec_scale
        point1 = start_keypoint + norm_vec
        point2 = start_keypoint - norm_vec
        point3 = end_keypoint + norm_vec
        point4 = end_keypoint - norm_vec
        if self.cfg.DEMO.DRAW_RIGID:
            drawer.line(tuple(point1) + tuple(point3), fill=color+(255, ), width=2)
            drawer.line(tuple(point2) + tuple(point4), fill=color+(255, ), width=2)
            drawer.line(tuple(point1) + tuple(point2), fill=color+(255, ), width=2)
            drawer.line(tuple(point3) + tuple(point4), fill=color+(255, ), width=2)
        part_width = 2 * (np.sqrt(np.sum(norm_vec ** 2)))
        return drawer, part_width

    def draw(self, image, human_bboxes, keypoints, human_scores, p_pasta, p_verb, human_ids, topk=5):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_shape = list(image.shape)

        # Construct a black sidebar.
        ones_shape = copy.deepcopy(im_shape)
        ones_shape[1] = 80
        image_ones = np.ones(ones_shape, dtype=image.dtype) * 0
        image = np.concatenate((image, image_ones), axis=1)
        
        pil_image = Image.fromarray(image).convert('RGBA')
        
        # White rectangle as bottom.
        overlay = Image.new('RGBA', pil_image.size, WHITE+(0,))
        overlay_draw = ImageDraw.Draw(overlay)
        
        canvas = Image.new('RGBA', pil_image.size, WHITE+(0,))
        draw = ImageDraw.Draw(canvas)

        font = ImageFont.truetype(self.cfg.DEMO.FONT_PATH, self.cfg.DEMO.FONT_SIZE)
        font_id = ImageFont.truetype(self.cfg.DEMO.FONT_PATH, self.cfg.DEMO.FONT_SIZE)

        draw.text((im_shape[1]+1, 3), '────────', font=font, fill=CYAN+(255, ))

        if human_bboxes is not None:
            human_count = 0
            extra_offset = self.cfg.DEMO.FONT_SIZE

            for idx in range(len(human_bboxes)):

                # Get meta information.
                human_bbox = human_bboxes[idx]
                human_keypoints = keypoints[idx]
                pasta_scores = p_pasta[idx]
                verb_scores = p_verb[idx]    
                ori_verb_scores = copy.deepcopy(verb_scores)
                
                # Get verb and pasta names to draw.
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
                for pasta_range_idx, pasta_range in enumerate(self.pasta_ranges):
                    pasta_sub_scores = pasta_scores[pasta_range[0]:pasta_range[1]]
                    pasta_names = self.pasta_name_list[pasta_range[0]:pasta_range[1]]

                    # rules for foot, hand and head pasta
                    if self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'hip' and self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'leg' and self.cfg.DATA.PASTA_NAMES[pasta_range_idx] != 'arm':
                        pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1]
                        for pasta_sub_idx in pasta_sub_idxs:
                            pasta_name = pasta_names[pasta_sub_idx]
                            if 'no_interaction' not in pasta_name:
                                pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                                break
                    
                    # rules for arm pasta
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

                    # rules for leg pasta  
                    elif self.cfg.DATA.PASTA_NAMES[pasta_range_idx] == 'leg':
                        pasta_sub_idxs = np.argsort(pasta_sub_scores)[::-1]
                        for pasta_sub_idx in pasta_sub_idxs:
                            pasta_name = pasta_names[pasta_sub_idx]
                            if 'no_interaction' not in pasta_name:
                                if 'is close with' not in pasta_name:
                                    pasta_draw_names.append(pasta_name.split(':')[-1].strip())
                                else:
                                    pasta_draw_names.append('')
                                break

                    # rules for hip pasta
                    else:
                        # print(human_count, pasta_sub_scores[:3])
                        if max(pasta_sub_scores[:3]) < 0.17:
                            pasta_draw_names.append('')
                        else:
                            pasta_draw_names.append('sit')
                
                # rules between pastas
                if pasta_draw_names[0] == 'stand' and pasta_draw_names[1] == 'jump' and 'jump' not in verb_draw_names[:3]:
                    pasta_draw_names[1] = ''
                if pasta_draw_names[1] == 'walk':
                    pasta_draw_names[2] = ''
                if pasta_draw_names[1] == 'is close with':
                    pasta_draw_names[1] = ''
                if pasta_draw_names[4] == 'be close to':
                    pasta_draw_names[4] = ''
                if pasta_draw_names[2] == 'sit' and pasta_draw_names[1] == 'jump':
                    pasta_draw_names[1] = ''
                if pasta_draw_names[2] == 'sit' and pasta_draw_names[1] == 'walk':
                    pasta_draw_names[1] = ''
                if pasta_draw_names[2] == 'sit' and pasta_draw_names[0] == 'jump':
                    pasta_draw_names[0] = 'stand'
                if pasta_draw_names[2] == 'sit' and pasta_draw_names[0] == 'walk':
                    pasta_draw_names[0] = 'stand'

                # Draw human box and human id box.
                draw.rectangle([(int(human_bbox[0]), int(human_bbox[1])), (int(human_bbox[2]),int(human_bbox[3]))], fill=None, outline=BLUE+(255, ), width=2)
                
                draw_id = human_count
                # draw_id = human_ids[idx] if self.mode == 'video' else human_count

                draw.rectangle([(int(human_bbox[0]), int(human_bbox[1])), (int(human_bbox[0]+1.5 * self.cfg.DEMO.FONT_SIZE / 1.7),int(human_bbox[1]+self.cfg.DEMO.FONT_SIZE+3))], fill=ID_COLOR_LIST[draw_id % 5]+(255,))
                
                draw.text((int(human_bbox[0])+3, int(human_bbox[1])+3), str(draw_id), font=font_id, fill=WHITE+(255, ))

                # Update sidebar.
                draw.text((im_shape[1]+1, 3+extra_offset), 'ID: '+str(draw_id), font=font, fill=CYAN+(255, ))  # +' {:.3f}'.format(verb_scores[verb_idx])
                extra_offset += self.cfg.DEMO.FONT_SIZE
                for draw_name in verb_draw_names:
                    draw.text((im_shape[1]+1, 3+extra_offset), draw_name, font=font, fill=GREEN+(255, ))  # +' {:.3f}'.format(verb_scores[verb_idx])
                    extra_offset += self.cfg.DEMO.FONT_SIZE
                draw.text((im_shape[1]+1, 3+extra_offset), '────────', font=font, fill=CYAN+(255, ))
                extra_offset += self.cfg.DEMO.FONT_SIZE
                
                # Draw the rigid of human body.
                foot_and_hand = dict()
                no_foot_and_hand = dict()
                lfoot_width = None
                lhand_width = None
                rfoot_width = None
                rhand_width = None
                head_width = None
                for part_name, kps_idxs in self.skeleton_to_parts.items():
                    if 'foot' in part_name or 'hand' in part_name:
                        foot_and_hand[part_name] = kps_idxs
                    else:
                        no_foot_and_hand[part_name] = kps_idxs
        
                for part_name, kps_idxs in no_foot_and_hand.items():
                    if part_name[1:] in self.pasta_name_dict:
                        part_idx = self.pasta_name_dict[part_name[1:]]

                    elif part_name in self.pasta_name_dict:
                        part_idx = self.pasta_name_dict[part_name]
                        
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
                            if score < 0.3:
                                continue

                            norm_vec_scale = 4.0 if 'hip' in self.cfg.DATA.PASTA_NAMES[part_idx] else 1.0

                            # Draw leg, hip and arm boxes
                            draw, part_width = self.rotated_rectangle(draw, start_keypoint, end_keypoint, PART_COLOR_LIST[part_idx], norm_vec_scale)
                            
                            if pair_id == len(kps_idxs)-2:
                                draw_name = pasta_draw_names[part_idx]
                                
                                # Draw hip pasta

                                if 'hip' in self.cfg.DATA.PASTA_NAMES[part_idx]:
                                    overlay_draw.rectangle([(int((start_keypoint[0]+end_keypoint[0])/2-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int((start_keypoint[1]+end_keypoint[1])/2-self.cfg.DEMO.FONT_SIZE/2)), (int((start_keypoint[0]+end_keypoint[0])/2+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int((start_keypoint[1]+end_keypoint[1])/2+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                                    draw.text((((start_keypoint[0]+end_keypoint[0])/2-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int((start_keypoint[1]+end_keypoint[1])/2-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))

                                # Draw leg and arm pasta
                                else:
                                    overlay_draw.rectangle([(int(start_keypoint[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(start_keypoint[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(start_keypoint[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(start_keypoint[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                                    draw.text((int(start_keypoint[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(start_keypoint[1])-self.cfg.DEMO.FONT_SIZE/2), draw_name, font=font, fill=BLUE+(255, ))
                                if part_name == 'lleg':
                                    lfoot_width = part_width
                                if part_name == 'rleg':
                                    rfoot_width = part_width
                                if part_name == 'larm':
                                    lhand_width = part_width
                                if part_name == 'rarm':
                                    rhand_width = part_width
                                
                    elif 'head' in self.cfg.DATA.PASTA_NAMES[part_idx]:
                        # Draw head box and pasta
                        head_keypoints = human_keypoints[kps_idxs]
                        x1, x2 = head_keypoints.min(axis=0)[0], head_keypoints.max(axis=0)[0]
                        head_width = x2-x1
                        head_centre = head_keypoints[0][:2]
                        head_score = head_keypoints[0][2]
                        if head_score < 0.3:
                            continue
                        draw_name = pasta_draw_names[part_idx]
                        draw.rectangle([(int(head_centre[0]-head_width/2), int(head_centre[1]-head_width/2)), (int(head_centre[0]+head_width/2), int(head_centre[1]+head_width/2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                        overlay_draw.rectangle([(int(head_centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(head_centre[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(head_centre[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(head_centre[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                        draw.text((int(head_centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(head_centre[1]-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))
                        
                for part_name, kps_idxs in foot_and_hand.items():
                    if part_name[1:] in self.pasta_name_dict:
                        part_idx = self.pasta_name_dict[part_name[1:]]

                    elif part_name in self.pasta_name_dict:
                        part_idx = self.pasta_name_dict[part_name]
                        
                    else:
                        raise NotImplementedError
                    draw_name = pasta_draw_names[part_idx]
                    kps_idx = kps_idxs[0]
                    centre = human_keypoints[kps_idx][:2]
                    score = human_keypoints[kps_idx][2]
                    if score < 0.3:
                        continue

                    # Draw foot and hand boxes and pasta
                    if 'rfoot' in part_name and rfoot_width is not None:
                        draw.rectangle([(int(centre[0]-rfoot_width*2), int(centre[1]-rfoot_width*2)), (int(centre[0]+rfoot_width*2), int(centre[1]+rfoot_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'lfoot' in part_name and lfoot_width is not None:
                        draw.rectangle([(int(centre[0]-lfoot_width*2), int(centre[1]-lfoot_width*2)), (int(centre[0]+lfoot_width*2), int(centre[1]+lfoot_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'rhand' in part_name and rhand_width is not None:
                        draw.rectangle([(int(centre[0]-rhand_width*2), int(centre[1]-rhand_width*2)), (int(centre[0]+rhand_width*2), int(centre[1]+rhand_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)
                    if 'lhand' in part_name and lhand_width is not None:
                        draw.rectangle([(int(centre[0]-lhand_width*2), int(centre[1]-lhand_width*2)), (int(centre[0]+lhand_width*2), int(centre[1]+lhand_width*2))], fill=None, outline=PART_COLOR_LIST[part_idx]+(255, ), width=2)

                    overlay_draw.rectangle([(int(centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(centre[1]-self.cfg.DEMO.FONT_SIZE/2)), (int(centre[0]+len(draw_name)*self.cfg.DEMO.FONT_SIZE/4), int(centre[1]+self.cfg.DEMO.FONT_SIZE/2))], fill=WHITE+(128, ))
                    draw.text((int(centre[0]-len(draw_name)*self.cfg.DEMO.FONT_SIZE/3), int(centre[1]-self.cfg.DEMO.FONT_SIZE/2)), draw_name, font=font, fill=BLUE+(255, ))

                # Draw remained skeleton
                draw, _ = self.rotated_rectangle(draw, human_keypoints[5][:2], human_keypoints[6][:2], PURPLE, 1)
                draw, _ = self.rotated_rectangle(draw, (human_keypoints[5][:2] + human_keypoints[6][:2]) / 2, (human_keypoints[11][:2] + human_keypoints[12][:2]) / 2, SOMECOLOR, 2/3)
                human_count += 1

        # Combine image and canvas
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = Image.alpha_composite(pil_image, canvas)
        pil_image = pil_image.convert('RGB')
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return cv2_image