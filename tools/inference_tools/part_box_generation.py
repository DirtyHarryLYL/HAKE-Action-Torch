import numpy as np

def map_17_to_16(joint_17):
    joint_16 = np.zeros((16, 3), dtype=np.float32)
    dict_map = {0:16, 1:14, 2:12, 3:11, 4:13, 5:15, 6:[11, 12], 7:[5,6], 9:[1,2], 10:10, 11:8, 12:6, 13:5, 14:7, 15:9}
    for idx in range(16):
        if idx == 8:
            continue # deal thrx joint later
        elif idx in [6, 7, 9]:
            #calc Pelv joint from two hip joint
            #calc neck joint from two shoulder joint
            #calc head joint from two eye joint
            joint_16[idx, 0] = (joint_17[dict_map[idx][0], 0] + joint_17[dict_map[idx][1], 0]) * 0.5
            joint_16[idx, 1] = (joint_17[dict_map[idx][0], 1] + joint_17[dict_map[idx][1], 1]) * 0.5
            joint_16[idx, 2] = (joint_17[dict_map[idx][0], 2] + joint_17[dict_map[idx][1], 2]) * 0.5
        else:
            joint_16[idx] = joint_17[dict_map[idx]]
    #calc thrx joint from head joint and neck, assume the distance is 1:3
    joint_16[8, 0] = joint_16[7, 0] * 0.75  + joint_16[9, 0] * 0.25
    joint_16[8, 1] = joint_16[7, 1] * 0.75  + joint_16[9, 1] * 0.25
    joint_16[8, 2] = joint_16[7, 2] * 0.75  + joint_16[9, 2] * 0.25

    return joint_16

def output_part_box(joint, img_bbox):
    
    flag_bad_joint = 0

    # 16 part names correspond to the center of 16 input joint
    part_size = [1.2, 1, 1, 1.2, 1.2, 1.2, 0.9, 1, 1, 0.9]
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]

    height = get_distance(joint, 6, 8)

    if (joint[8, 2] < 0.2) or (joint[6, 2] < 0.2):
        flag_bad_joint = 1

    group_score_head      = (joint[7, 2] + joint[8, 2] + joint[9, 2]) / 3
    group_score_left_arm  = (joint[13, 2] + joint[14, 2] + joint[15, 2]) / 3
    group_score_right_arm = (joint[10, 2] + joint[11, 2] + joint[12, 2]) / 3
    group_score_left_leg  = (joint[3, 2] + joint[4, 2] + joint[5, 2]) / 3
    group_score_right_leg = (joint[0, 2] + joint[1, 2] + joint[2, 2]) / 3
    
    # 'Pelv'&'Neck' scaling by the distance of Pelv and Neck
    bbox = np.zeros((10, 4), dtype=np.float32)
    for i in range(10):
        score_joint = joint[part[i], 2]

        # the keypoint is not reliable/ cannot be seen / do not exist
        if (score_joint < 0.2):
            bbox[i, 0] = img_bbox[0]
            bbox[i, 1] = img_bbox[1]
            bbox[i, 2] = img_bbox[2]
            bbox[i, 3] = img_bbox[3]

        # the keypoint is reliable, but the distance cannot be measured by distance between pelv and neck
        elif (score_joint >= 0.2) and (flag_bad_joint == 1):
            if i == 5: # head group
                if group_score_head > 0.2:
                    half_box_width = get_distance(joint, 7, 9)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [6,7]: # right arm group
                if group_score_right_arm > 0.2: 
                    half_box_width = get_distance(joint, 10, 12)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [8,9]: # left arm group
                if group_score_left_arm > 0.2: 
                    half_box_width = get_distance(joint, 13, 15)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [0,1]: # right leg group
                if group_score_right_leg > 0.2: 
                    half_box_width = get_distance(joint, 0, 2)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            elif i in [2,3]: # left leg group
                if group_score_left_leg > 0.2: 
                    half_box_width = get_distance(joint, 3, 5)
                    pbox = get_part_box(i, joint, half_box_width)
                    bbox[i, 0] = pbox[0]
                    bbox[i, 1] = pbox[1]
                    bbox[i, 2] = pbox[2]
                    bbox[i, 3] = pbox[3]
                else:
                    bbox[i, 0] = img_bbox[0]
                    bbox[i, 1] = img_bbox[1]
                    bbox[i, 2] = img_bbox[2]
                    bbox[i, 3] = img_bbox[3]
            else: # pelv keypoint
                bbox[i, 0] = img_bbox[0]
                bbox[i, 1] = img_bbox[1]
                bbox[i, 2] = img_bbox[2]
                bbox[i, 3] = img_bbox[3]
    
        else: # the keypoint is reliable and the distance can be measured by distance between pelv and neck
            half_box_width = height * part_size[i] / 2
            pbox = get_part_box(i, joint, half_box_width)
            bbox[i, 0] = pbox[0]
            bbox[i, 1] = pbox[1]
            bbox[i, 2] = pbox[2]
            bbox[i, 3] = pbox[3]
    return np.concatenate([np.zeros((10, 1)), bbox], axis=1)

def get_part_box(i, joint, half_box_width):
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]
    center_x = joint[part[i], 0]
    center_y = joint[part[i], 1]
    return center_x - half_box_width, center_y - half_box_width, center_x + half_box_width, center_y + half_box_width

def get_distance(joint, keypoint1, keypoint2):
    height_y = joint[keypoint1, 1] - joint[keypoint2, 1] 
    height_x = joint[keypoint1, 0] - joint[keypoint2, 0]
    return np.sqrt(height_x ** 2 + height_y ** 2)

def check_iou(human_bbox_pkl, human_bbox_pose):

    x1, y1, x2, y2 = human_bbox_pose
    x1d, y1d, x2d, y2d = human_bbox_pkl

    xa = max(x1, x1d)
    ya = max(y1, y1d)
    xb = min(x2, x2d)
    yb = min(y2, y2d)

    iw1 = xb - xa + 1
    iw2 = yb - ya + 1

    if iw1 > 0 and iw2 > 0:
        inter_area = iw1 * iw2
        a_area = (x2 - x1) * (y2 - y1)
        b_area = (x2d - x1d) * (y2d - y1d)
        union_area = a_area + b_area - inter_area
        return inter_area / float(union_area)
    else:
        return 0

if __name__ == '__main__':
    # Given the human bounding box and the corresponding human pose, Generate the correponding part box
    hbox = np.zeros((4))
    joint_17 = np.zeros((17, 3))               # the pose
    joint_16 = map_17_to_16(joint_17)
    part_box = output_part_box(joint_16, hbox) # return the part bounding box
