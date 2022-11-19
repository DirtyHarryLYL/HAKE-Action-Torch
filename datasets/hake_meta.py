
"""
    number of classes to classify it includes the "no_interaction"
"""
num_classes = {}
num_classes['foot'] = 16
num_classes['leg'] = 15
num_classes['hip'] = 6
num_classes['hand'] = 34
num_classes['arm'] = 8
num_classes['head'] = 14
num_classes['verb'] = 157
num_classes['pasta_binary'] = 6  # use to classify whether exist a state for a specific part

"""
    ignore idx indicate the "no_interaction", 0-based
    for part, it is always the last one
    for verbs, it is the 57-th
    for pasta_binary, there is no "no_interaction"
"""
ignore_idx = {}
ignore_idx['foot'] = num_classes['foot'] - 1
ignore_idx['leg'] = num_classes['leg'] - 1
ignore_idx['hip'] = num_classes['hip'] - 1
ignore_idx['hand'] = num_classes['hand'] - 1
ignore_idx['arm'] = num_classes['arm'] - 1
ignore_idx['head'] = num_classes['head'] - 1
ignore_idx['verb'] = 57
ignore_idx['pasta_binary'] = -1  # < 0 means there is no idx for no_interaction


part_names = ['foot', 'leg', 'hip', 'hand', 'arm', 'head', 'verb']
num_part = len(part_names)

class_names = {}
class_names['verb'] = ["adjust","assemble","block","blow","board","break","brush_with","buy","carry","catch","chase","check","clean","control","cook","cut","cut_with","direct","drag","dribble","drink_with","drive","dry","eat","eat_at","exit","feed","fill","flip","flush","fly","greet","grind","groom","herd","hit","hold","hop_on","hose","hug","hunt","inspect","install","jump","kick","kiss","lasso","launch","lick","lie_on","lift","light","load","lose","make","milk","move","no_interaction","open","operate","pack","paint","park","pay","peel","pet","pick","pick_up","point","pour","pull","push","race","read","release","repair","ride","row","run","sail","scratch","serve","set","shear","sign","sip","sit_at","sit_on","slide","smell","spin","squeeze","stab","stand_on","stand_under","stick","stir","stop_at","straddle","swing","tag","talk_on","teach","text_on","throw","tie","toast","train","turn","type_on","walk","wash","watch","wave","wear","wield","zip","bend/bow (at the waist)","crawl","crouch/kneel","dance","fall down","get up","martial art","swim","chop","close (e.g., a door, a box)","dig","shovel","climb (e.g., a mountain)","dress/put on clothing","enter","fishing","play board game","play musical instrument","play with pets","play with kids","extract","press","put down","shoot","smoke","clink glass","hand clap","hand shake","take a photo","touch (an object)","work on a computer","write","fight/hit (a person)","give/serve (an object) to (a person)","take (an object) from (a person)","grab (a person)","listen to (a person)","listen (e.g., to music)","sing to (e.g., self, a person, a group)","talk to (e.g., self, a person, a group)"]
class_names['foot'] = ["foot: stand on","foot: tread on","foot: walk with","foot: walk to","foot: run with","foot: run to","foot: dribble","foot: kick","foot: jump down","foot: jump with","foot: walk away","foot: crawl","foot: dance","foot: fall down","foot: martial art","foot: no_interaction"]
class_names['leg'] = ["leg: walk with","leg: walk to","leg: run with","leg: run to","leg: jump with","leg: is close with","leg: straddle","leg: jump down","leg: walk away","leg: bend","leg: kneel","leg: crawl","leg: dance","leg: martial art","leg: no_interaction"]
class_names['hip'] = ["hip: sit on","hip: sit in","hip: sit beside","hip: be close with","hip: bend","hip: no_interaction"]
class_names['hand'] = ["hand: hold","hand: carry","hand: reach for","hand: touch","hand: put on","hand: twist","hand: wear","hand: throw","hand: throw out","hand: write on","hand: point with","hand: point to","hand: use something to point to","hand: press","hand: squeeze","hand: scratch","hand: pinch","hand: gesture to","hand: push","hand: pull","hand: pull with something","hand: wash","hand: wash with something","hand: hold in both hands","hand: lift","hand: raise(over head)","hand: feed","hand: cut with something","hand: catch with something","hand: pour into","hand: crawl","hand: dance","hand: martial art","hand: no_interaction"]
class_names['arm'] = ["arm: shoulder carry","arm: be close to","arm: hug","arm: swing","arm: crawl","arm: dance","arm: martial art","arm: no_interaction"]
class_names['head'] = ["head: eat","head: inspect","head: talk with something","head: talk to","head: be close with","head: kiss","head: put something over head","head: lick","head: blow","head: drink with","head: smell","head: wear","head: listen","head: no_interaction"]
class_names['pasta'] = ["foot: stand on","foot: tread on","foot: walk with","foot: walk to","foot: run with","foot: run to","foot: dribble","foot: kick","foot: jump down","foot: jump with","foot: walk away","foot: crawl","foot: dance","foot: fall down","foot: martial art","foot: no_interaction","leg: walk with","leg: walk to","leg: run with","leg: run to","leg: jump with","leg: is close with","leg: straddle","leg: jump down","leg: walk away","leg: bend","leg: kneel","leg: crawl","leg: dance","leg: martial art","leg: no_interaction","hip: sit on","hip: sit in","hip: sit beside","hip: be close with","hip: bend","hip: no_interaction","hand: hold","hand: carry","hand: reach for","hand: touch","hand: put on","hand: twist","hand: wear","hand: throw","hand: throw out","hand: write on","hand: point with","hand: point to","hand: use something to point to","hand: press","hand: squeeze","hand: scratch","hand: pinch","hand: gesture to","hand: push","hand: pull","hand: pull with something","hand: wash","hand: wash with something","hand: hold in both hands","hand: lift","hand: raise(over head)","hand: feed","hand: cut with something","hand: catch with something","hand: pour into","hand: crawl","hand: dance","hand: martial art","hand: no_interaction","arm: shoulder carry","arm: be close to","arm: hug","arm: swing","arm: crawl","arm: dance","arm: martial art","arm: no_interaction","head: eat","head: inspect","head: talk with something","head: talk to","head: be close with","head: kiss","head: put something over head","head: lick","head: blow","head: drink with","head: smell","head: wear","head: listen","head: no_interaction"]
class_names['pasta_binary'] = ["foot","leg","hip","hand","arm","head"]



