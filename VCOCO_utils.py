import os
import pickle
import numpy as np
from vsrl_eval_output_txt import VCOCOeval

mapping = {
    "hold_obj":   (0, 0), 
    "sit_instr":  (2, 0), 
    "ride_instr": (3, 0), 
    "look_obj":   (5, 0), 
    "hit_instr":  (6, 0), 
    "hit_obj":    (6, 1), 
    "eat_obj":    (7, 0), 
    "eat_instr":  (7, 1), 
    "jump_instr": (8, 0), 
    "lay_instr":  (9, 0), 
    "talk_on_phone_instr": (10, 0), 
    "carry_obj":  (11, 0), 
    "throw_obj":  (12, 0), 
    "catch_obj":  (13, 0), 
    "cut_instr":  (14, 0), 
    "cut_obj":    (14, 1), 
    "work_on_computer_instr": (16, 0), 
    "ski_instr":  (17, 0), 
    "surf_instr": (18, 0), 
    "skateboard_instr": (19, 0), 
    "drink_instr":      (21, 0), 
    "kick_obj":         (22, 0), 
    "point_instr":      (23, 0), 
    "read_obj":         (24, 0), 
    "snowboard_instr":  (25, 0)
}

TIN_index = {
    "surf_instr": 0, "ski_instr": 1, "cut_instr": 2, "walk": 3, "cut_obj": 4, "ride_instr": 5, 
    "talk_on_phone_instr": 6, "kick_obj": 7, "work_on_computer_instr": 8, "eat_obj": 9, "sit_instr": 10, 
    "jump_instr": 11, "lay_instr": 12, "drink_instr": 13, "carry_obj": 14, "throw_obj": 15, 
    "eat_instr": 16, "smile": 17, "look_obj": 18, "hit_instr": 19, "hit_obj": 20, "snowboard_instr": 21, 
    "run": 22, "point_instr": 23, "read_obj": 24, "hold_obj": 25, "skateboard_instr": 26, "stand": 27, "catch_obj": 28}

TIN2VSG = [14, 28, 2, 4, 13, 16, 9, 19, 20, 25, 11, 7, 12, 18, 23, 24, 5, 22, 10, 26, 1, 17, 21, 27, 0, 6, 15, 3, 8]

NUM_VERBS = 29
VERB2ID = {
    u'carry': 0, u'catch': 1, u'cut_instr': 2, u'cut_obj': 3, u'drink': 4, u'eat_instr': 5, u'eat_obj': 6,
    u'hit_instr': 7, u'hit_obj': 8, u'hold': 9, u'jump': 10, u'kick': 11, u'lay': 12, u'look': 13, u'point': 14,
    u'read': 15, u'ride': 16, u'run': 17, u'sit': 18, u'skateboard': 19, u'ski': 20, u'smile': 21, u'snowboard': 22,
    u'stand': 23, u'surf': 24, u'talk_on_phone': 25, u'throw': 26, u'walk': 27, u'work_on_computer': 28
}
ID2VERB = {v: k for k, v in VERB2ID.items()}
FULL_VERBS = {
    'carry_agent', 'carry_obj', 'catch_agent', 'catch_obj', 'cut_agent', 'cut_obj', 'cut_agent', 'cut_instr',
    'drink_agent', 'drink_instr', 'eat_agent', 'eat_obj', 'eat_agent', 'eat_instr', 'hit_agent', 'hit_obj', 'hit_agent',
    'hit_instr', 'hold_agent', 'hold_obj', 'jump_agent', 'jump_instr', 'kick_agent', 'kick_obj', 'lay_agent', 'lay_instr',
    'look_agent', 'look_obj', 'point_agent', 'point_instr', 'read_agent', 'read_obj', 'ride_agent', 'ride_instr',
    'run_agent', 'sit_agent', 'sit_instr', 'skateboard_agent', 'skateboard_instr', 'ski_agent', 'ski_instr', 'smile_agent',
    'snowboard_agent', 'snowboard_instr', 'stand_agent', 'surf_agent', 'surf_instr', 'talk_on_phone_agent',
    'talk_on_phone_instr', 'throw_agent', 'throw_obj', 'walk_agent', 'work_on_computer_agent', 'work_on_computer_instr'
}

VERB2AGENT = {verb:verb.split('_obj')[0].split('_instr')[0] + '_agent' for verb in VERB2ID}
VERB2NON_AGENT = {}
for full_verb in FULL_VERBS:
    if '_agent' in full_verb:
        continue
    if full_verb in VERB2ID:
        VERB2NON_AGENT[full_verb] = full_verb
    else:
        VERB2NON_AGENT['_'.join(full_verb.split('_')[:-1])] = full_verb

EMPTY_VERBS = {k: 0.0 for k in VERB2AGENT.values()}
EMPTY_VERBS.update({k: [np.nan, np.nan, np.nan, np.nan, 0.0] for k in VERB2NON_AGENT.values()})
with open('prior.pkl', 'rb') as f:
    PRIOR = pickle.load(f, encoding='latin1')

DATA_DIR = '/Disk1/yonglu/iCAN/Data'
vcocoeval = VCOCOeval(
    DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
    DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
    DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids') 

def apply_prior(Object_class, prediction):
    if Object_class != 0:
        prediction *= PRIOR[Object_class].squeeze(0)
    return prediction


def get_map_vcoco(
    keys,
    sub_ids,
    act_scores,
    spatial,
    obj_classes,
    DATA_DIR='/Disk2/iCAN/Data'
):

    output_file = 'vcoco_results' + str(np.random.randint(10000)).zfill(5)

    output = {} # VCOCO format pickle dict

    h_bbox = spatial[:, :4]
    o_bbox = spatial[:, 4:]

    for pair_idx, _ in enumerate(keys):
        human_key = (keys[pair_idx], sub_ids[pair_idx])
        if human_key not in output:
            # Initalize a human in output
            output[human_key] = {
                'image_id': keys[pair_idx],
                'person_box': h_bbox[pair_idx]
            }
            output[human_key].update(EMPTY_VERBS)
        
        # Apply prior
        action_score = apply_prior(obj_classes[pair_idx], act_scores[pair_idx])

        for verb_id in np.where(action_score > 0)[0]:
            verb = ID2VERB[verb_id]
            score = action_score[verb_id]
            # Update agent action
            agent_verb = VERB2AGENT[verb]
            output[human_key][agent_verb] = max(output[human_key][agent_verb], score)

            # Skip none agent action if there is no object in the image
            # if obj_classes[pair_idx] == 0 or verb not in VERB2NON_AGENT:
            #     continue
            # cur_o_bbox = o_bbox[pair_idx, :].tolist()

            # Set the bbox to none if object class is 0, i.e. no object
            if verb not in VERB2NON_AGENT:
                continue
            if obj_classes[pair_idx] == 0:
                cur_o_bbox = [np.nan, np.nan, np.nan, np.nan]
            else:
                cur_o_bbox = o_bbox[pair_idx, :].tolist()

                
            non_agent_verb = VERB2NON_AGENT[verb]
            # Update object and score
            if score > output[human_key][non_agent_verb][-1]:
                output[human_key][non_agent_verb] = cur_o_bbox + [score]
    
    output = list(output.values())
        
    with open(output_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    agent, role1, role2 = vcocoeval._do_eval(output_file, ovr_thresh = 0.5)
    os.remove(output_file)

    return agent, role1, role2



def get_map_vcoco_no_save(
    keys,
    sub_ids,
    act_scores,
    spatial,
    obj_classes,
    DATA_DIR='/Disk2/iCAN/Data'
):

    output = {} # VCOCO format pickle dict

    h_bbox = spatial[:, :4]
    o_bbox = spatial[:, 4:]

    for pair_idx, _ in enumerate(keys):
        human_key = (keys[pair_idx], sub_ids[pair_idx])
        if human_key not in output:
            # Initalize a human in output
            output[human_key] = {
                'image_id': keys[pair_idx],
                'person_box': h_bbox[pair_idx]
            }
            output[human_key].update(EMPTY_VERBS)
        
        # Apply prior
        action_score = apply_prior(obj_classes[pair_idx], act_scores[pair_idx])

        for verb_id in np.where(action_score > 0)[0]:
            verb = ID2VERB[verb_id]
            score = action_score[verb_id]
            # Update agent action
            agent_verb = VERB2AGENT[verb]
            output[human_key][agent_verb] = max(output[human_key][agent_verb], score)

            # Skip none agent action if there is no object in the image
            # if obj_classes[pair_idx] == 0 or verb not in VERB2NON_AGENT:
            #     continue
            # cur_o_bbox = o_bbox[pair_idx, :].tolist()

            # Set the bbox to none if object class is 0, i.e. no object
            if verb not in VERB2NON_AGENT:
                continue
            if obj_classes[pair_idx] == 0:
                cur_o_bbox = [np.nan, np.nan, np.nan, np.nan]
            else:
                cur_o_bbox = o_bbox[pair_idx, :].tolist()

                
            non_agent_verb = VERB2NON_AGENT[verb]
            # Update object and score
            if score > output[human_key][non_agent_verb][-1]:
                output[human_key][non_agent_verb] = cur_o_bbox + [score]
    
    output = list(output.values())

    res = vcocoeval._do_eval(output, ovr_thresh = 0.5)

    return res
