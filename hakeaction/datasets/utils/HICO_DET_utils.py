# --------------------------------------------------------
# hakeaction
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import pickle
import random
import cv2
import h5py
import os
import sys
import inspect
import trimesh
from sympy import sympify
from random import randint
from .hico_generate_utils import *
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

rare_index = np.array([ 9, 23,28, 45,51, 56,63, 64,67, 71,77, 78,81, 84,85, 91,100,101,105,108,113,128,136,137,150,159,166,167,169,173,180,182,185,189,190,193,196,199,206,207,215,217,223,228,230,239,240,255,256,258,261,262,263,275,280,281,282,287,290,293,304,312,316,318,326,329,334,335,346,351,352,355,359,365,380,382,390,391,392,396,398,399,400,402,403,404,405,406,408,411,417,419,427,428,430,432,437,440,441,450,452,464,470,475,483,486,499,500,505,510,515,518,521,523,527,532,536,540,547,548,549,550,551,552,553,556,557,561,579,581,582,587,593,594,596,597,598,600,]) - 1
rare = np.zeros(600)
rare[rare_index] += 2

obj_range = [
    (161, 170), (11, 24),   (66, 76),   (147, 160), (1, 10), 
    (55, 65),   (187, 194), (568, 576), (32, 46),   (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86),   (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92),   (93, 96),   (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]


hoi_no_inter_all = [
    10,24,31,46,54,65,76,86,92,96,107,111,129,146,160,170,174,186,194,198,208,214,
    224,232,235,239,243,247,252,257,264,273,283,290,295,305,313,325,330,336,342,348,
    352,356,363,368,376,383,389,393,397,407,414,418,429,434,438,445,449,453,463,474,
    483,488,502,506,516,528,533,538,546,550,558,562,567,576,584,588,595,600
]

list_no_inter = [10,24,31,46,54,65,76,86,92,96,107,111,
                    129,146,160,170,174,186,194,198,208,214,224,232,235,239,
                    243,247,252,257,264,273,283,290,295,305,313,325,330,336,
                    342,348,352,356,363,368,376,383,389,393,397,407,414,418,
                    429,434,438,445,449,453,463,474,483,488,502,506,516,528,
                    533,538,546,550,558,562,567,576,584,588,595,600]

OBJECT_MASK = [
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
    ]

HO_weight = [
                9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423, 
                11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699, 
                6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912, 
                5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048, 
                8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585, 
                12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731, 
                7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116, 
                14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678, 
                12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933, 
                14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973, 
                12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967, 
                4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679, 
                9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917, 
                10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057, 
                8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799, 
                9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912, 
                9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591, 
                12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264, 
                11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264, 
                7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304, 
                10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384, 
                11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143, 
                11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264, 
                7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264, 
                8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909, 
                7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748, 
                14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135, 
                11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411, 
                14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533, 
                10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361, 
                9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222, 
                9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963, 
                8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324, 
                9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948, 
                5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264, 
                11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248, 
                10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862, 
                8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224, 
                12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973, 
                12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493, 
                14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973, 
                11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892, 
                10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825, 
                12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799, 
                9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505, 
                12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368, 
                7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925, 
                7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015, 
                7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551, 
                9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584, 
                5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826, 
                11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799, 
                10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059, 
                13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571, 
                11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501, 
                14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411, 
                7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305, 
                11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186, 
                12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264, 
                10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
            ]

binary_weight = [1.6094379124341003, 0.22314355131420976]


def bbox_trans(human_box_ori, object_box_ori, ratio, size = 64):

    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()
    
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]    

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # shift the top-left corner to (0,0)
    
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 

    if ratio == 'height': # height is larger than width
        
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        
        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1



        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else: # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        
        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        
        
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

 
    return np.round(human_box), np.round(object_box)

def Get_next_sp(human_box, object_box):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')
    
    Pattern = np.zeros((64,64,2))
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1


    return Pattern

def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def Augmented_box(bbox, shape, image_id, augment = 15, break_flag = True):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if break_flag == True and time_count > 150:
            return box
            
    return box

def Generate_action(action_list):
    action_ = np.zeros(29)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,29)
    return action_

def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_

def Generate_action_30(action_list):
    action_ = np.zeros(30)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,30)
    return action_

def draw_relation(human_pattern, joints, size = 64):

    joint_relation = [[1,3],[2,4],[0,1],[0,2],[0,17],[5,17],[6,17],[5,7],[6,8],[7,9],[8,10],[11,17],[12,17],[11,13],[12,14],[13,15],[14,16]]
    color = [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))
    
    # cv2.rectangle(skeleton, (int(human_pattern[0]), int(human_pattern[1])), (int(human_pattern[2]), int(human_pattern[3])), (255))
    # cv2.imshow("Joints", skeleton)
    # cv2.waitKey(0)
    # print(skeleton[:,:,0])

    return skeleton

def get_skeleton(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    return draw_relation(human_pattern, joints)

def get_skeleton_calAttMap(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    xmap = np.tile(np.arange(64).reshape(1, -1), [64, 1]).astype(np.float32)
    ymap = np.tile(np.arange(64).reshape(-1, 1), [1, 64]).astype(np.float32)

    att_map_all = []
    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        att_map = 1 + np.sqrt((joint_x - xmap) ** 2, (joint_y - ymap) ** 2)
        att_map = 1. / att_map
        att_map /= np.sum(att_map)
        att_map = att_map.reshape((1, 64, 64, 1))
        att_map_all.append(att_map)
    att_map_all = np.concatenate(att_map_all, axis=3)

    return draw_relation(human_pattern, joints), att_map_all

def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64,64,2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

    if human_pose != None and len(human_pose) == 51:
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64,64,1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05

    Pattern = np.concatenate((Pattern, skeleton), axis=2)

    return Pattern

def Get_next_sp_with_pose_calAttMap(human_box, object_box, human_pose, num_joints=17):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64,64,2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

    if human_pose != None and len(human_pose) == 51:
        skeleton, att_map = get_skeleton_calAttMap(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64,64,1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05
        att_map = np.zeros((1, 64, 64, 17))

    Pattern = np.concatenate((Pattern, skeleton), axis=2).reshape(1, 64, 64, 3)

    return Pattern, att_map

##############################################################################
# for vcoco with pose pattern
##############################################################################

def Get_Next_Instance_HO_Neg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, config):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes_solo']= Human_augmented_solo
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, Human_augmented_solo, action_H = [], [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        Human_augmented_solo.extend(Human_augmented_temp)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]
    print(Object)
    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)
    Human_augmented_solo.extend(Human_augmented_temp)
    
    print(Object_augmented_temp)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label

def Get_Next_Instance_HO_spNeg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, config):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['Hsp_boxes']   = Human_augmented_sp
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_sp'] = action_sp
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_sp']     = mask_sp
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]
    print(Object)

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    action_sp = np.array(action_HO).copy()

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label


##############################################################################
# for hico with pose map
##############################################################################

def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, config):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    # print("GT= ",GT)
    # GT_count = len(GT)
    GT_count = 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0]
    GT = [GT]
    Human_augmented, Object_augmented, action_HO = [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        # print("==> i={}, GT[i]={}".format(i,GT[i]))
        try:
            Human    = GT[0][2]
            Object   = GT[0][3]
        except:
            continue

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]
    # print(Object)
    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]
    # print(Object_augmented_temp)
    action_HO__temp = Generate_action_HICO(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    num_pos = len(Human_augmented)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 3) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label

def get_order_obj():
    obj_range = [
    (161, 170), (11, 24), (66, 76), (147, 160), (1, 10), 
    (55, 65), (187, 194), (568, 576), (32, 46), (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86), (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]
    f = open('./hakeaction/datasets/utils/hico_files/hico_list_hoi.txt','r')
    line = f.readline()
    line = f.readline()
    list_hoi = []
    list_hoi.append("None")
    line = f.readline()
    while line:
        tmp = line.strip('\n').split()
        list_hoi.append([tmp[1],tmp[2]])
        line = f.readline()

    obj_order_dict = {}
    order_obj_list = []
    order_obj_list.append(' ')
    for i in range(len(obj_range)):
        order_obj_list.append(list_hoi[obj_range[i][0]][0]) 
        obj_order_dict[order_obj_list[i+1]] = i + 1

    obj_para_dict = {}
    f = open('./hakeaction/datasets/utils/hico_files/hico_obj_parameter.txt','r')
    line = f.readline()
    cnt = 0
    while line:
        cnt = cnt + 1
        tmp = line.strip('\n').split()
        tmp_dict = {}
        tmp_dict['ratio'] = float(tmp[1])
        tmp_dict['gamma_min'] = float(tmp[2])
        tmp_dict['gamma_max'] = float(tmp[3])
        obj_para_dict[tmp[0]] = tmp_dict
        line = f.readline()
    
    return list_hoi, order_obj_list, obj_para_dict

def Get_Next_Instance_HO_Neg_HICO_3D(imgPath, GT, Trainval_Neg, image_id, Pos_augment, Neg_select, config, vertex_choice, smplx_model_data, pointNet=None):

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, smplx, att_2D_map, pc = \
        HO_Neg_HICO_3D(GT, Trainval_Neg, image_id, imgPath, Pos_augment, Neg_select, config, vertex_choice, smplx_model_data, pointNet)
    
    blobs = {}
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['smplx']       = smplx
    blobs['gt_class_HO'] = action_HO
    blobs['att_2D_map']  = att_2D_map
    blobs['pc']          = pc

    return blobs

def HO_Neg_HICO_3D(GT, Trainval_Neg, image_id, imgPath, Pos_augment, Neg_select, config, vertex_choice, model_data, pointNet=None):

    GT_count      = len(GT) # GT for specific img
    GT_idx        = list(np.random.choice(range(GT_count), Pos_augment))
    REQUIRE_SMPLX = config.DATA_REQUIRE_SMPLX
    REQUIRE_PC    = config.DATA_REQUIRE_PC
    Human_augmented, Object_augmented, action_HO, Pattern, smplx, att_2D_map, pc = [], [], [], [], [], [], []
    list_hoi, order_obj_list, obj_para_dict = get_order_obj()
    
    for i in GT_idx:
        Human     = np.array(GT[i][2], dtype='float64') # Human  BBox
        Object    = np.array(GT[i][3], dtype='float64') # Object BBox
        hoi_class = GT[i][1]                            # Ground Truth HOI Label
        Human_augmented.append(np.array([0, Human[0], Human[1], Human[2], Human[3]], dtype='float64').reshape(1, -1))
        Object_augmented.append(np.array([0, Object[0], Object[1], Object[2], Object[3]], dtype='float64').reshape(1, -1))
        action_HO.append(Generate_action_HICO(GT[i][1]))
        pat_tmp, att_tmp = Get_next_sp_with_pose_calAttMap(Human, Object, GT[i][5])
        Pattern.append(pat_tmp)
        att_2D_map.append(att_tmp)

        if not REQUIRE_SMPLX and not REQUIRE_PC:
            smplx.append(np.zeros((1, 85)))
            pc.append(np.zeros((1, 1228, 256)))
            continue
            
        if (GT[i][-1] != None) and os.path.exists(config.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, GT[i][-1]['id'])):
            # get result from GT: Trainval_GT_HICO_with_smplx.pkl
            result = GT[i][-1] 
            if REQUIRE_SMPLX:
                smplx.append(
                    np.concatenate([
                        result['left_hand_pose'], result['right_hand_pose'],
                        result['leye_pose'],      result['reye_pose'],       result['jaw_pose'], 
                        result['body_pose'],      result['expression'],      result['betas'],
                    ], axis=1))
            else:
                smplx.append(np.zeros((1, 85)))
            
            # Dynamic generation of pc
            if REQUIRE_PC:
                obj_name = list_hoi[hoi_class][0]
                tmp_pc   = getPointCloudFeature(pointNet=pointNet, cfg=config,  obj_name=obj_name,                 \
                                                    hbox=Human,      obox=Object,  imgPath=imgPath, key=image_id,    \
                                                    idx=GT[i][-1]['id'], model_data=model_data, smplx_result=result, \
                                                    vertex_choice=vertex_choice, obj_para_dict=obj_para_dict)
                pc.append(tmp_pc)
            else:
                pc.append(np.zeros((1, 1228, 256)))

        else:
            smplx.append(np.zeros((1, 85)))
            pc.append(np.zeros((1, 1228, 256)))
        
    num_pos = len(Human_augmented)
    if image_id in Trainval_Neg.keys():
        if len(Trainval_Neg[image_id]) < Neg_select:
            for i in range(len(Trainval_Neg[image_id])):
                Neg       = Trainval_Neg[image_id][i]
                hoi_class = Neg[1]
                Human     = Neg[2]
                Object    = Neg[3]
                obj_name  = list_hoi[hoi_class][0]
                Human_augmented.append(np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5))
                Object_augmented.append(np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5))
                action_HO.append(Generate_action_HICO([Neg[1]]))
                pat_tmp, att_tmp = Get_next_sp_with_pose_calAttMap(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[4])
                Pattern.append(pat_tmp)
                att_2D_map.append(att_tmp)

                if not REQUIRE_SMPLX and not REQUIRE_PC:
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 256)))
                    continue

                if (Neg[-1] != None) and os.path.exists(config.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1]['id'])):
                    result = Trainval_Neg[image_id][i][-1]
                    if REQUIRE_SMPLX:
                        smplx.append(np.concatenate([
                                            result['left_hand_pose'], result['right_hand_pose'],
                                            result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                            result['expression'], result['betas'],
                                        ], axis=1))
                    else:
                        smplx.append(np.zeros((1, 85)))

                    if REQUIRE_PC:
                        tmp_pc = getPointCloudFeature(pointNet=pointNet, cfg=config,   obj_name=obj_name,
                                                        hbox=Human,      obox=Object,  imgPath=imgPath,   key=image_id,
                                                        idx=Neg[-1]['id'], model_data=model_data, smplx_result=result,
                                                        vertex_choice=vertex_choice, obj_para_dict=obj_para_dict)
                        pc.append(tmp_pc)   
                    else:
                        pc.append(np.zeros((1, 1228, 256)))             
                
                else:    
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 256)))
        else:

            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg       = Trainval_Neg[image_id][List[i]]
                hoi_class = Neg[1]
                Human     = Neg[2]
                Object    = Neg[3]
                obj_name  = list_hoi[hoi_class][0]
                Human_augmented.append(np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5))
                Object_augmented.append(np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5))
                action_HO.append(Generate_action_HICO([Neg[1]]))
                pat_tmp, att_tmp = Get_next_sp_with_pose_calAttMap(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[4])
                Pattern.append(pat_tmp)
                att_2D_map.append(att_tmp)
                
                if not REQUIRE_SMPLX and not REQUIRE_PC:
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 256)))
                    continue

                if (Neg[-1] != None) and os.path.exists(config.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1]['id'])):
                    result = Trainval_Neg[image_id][List[i]][-1]
                    if REQUIRE_SMPLX:
                        smplx.append(np.concatenate([
                                            result['left_hand_pose'], result['right_hand_pose'],
                                            result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                            result['expression'], result['betas'],
                                        ], axis=1))
                    else:
                        smplx.append(np.zeros((1, 85)))

                    if REQUIRE_PC:
                        tmp_pc = getPointCloudFeature(pointNet=pointNet, cfg=config,   obj_name=obj_name,
                                                        hbox=Human,      obox=Object,  imgPath=imgPath,   key=image_id,
                                                        idx=Neg[-1]['id'], model_data=model_data, smplx_result=result,
                                                        vertex_choice=vertex_choice, obj_para_dict=obj_para_dict)
                        pc.append(tmp_pc)
                    else:
                        pc.append(np.zeros((1, 1228, 256))) 

                else:
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 256)))
    
    Pattern          = np.concatenate(Pattern, axis=0)
    Human_augmented  = np.concatenate(Human_augmented, axis=0)
    Object_augmented = np.concatenate(Object_augmented, axis=0)
    action_HO        = np.concatenate(action_HO, axis=0)
    smplx            = np.concatenate(smplx, axis=0)
    att_2D_map       = np.concatenate(att_2D_map, axis=0)
    pc               = np.concatenate(pc, axis=0)

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, smplx, att_2D_map, pc

def get_joints(model_data, vertices):
    '''
    if (args.gender == 'neutral'):
        suffix = 'SMPLX_NEUTRAL.pkl'
    elif (args.gender == 'male'):
        suffix = 'SMPLX_MALE.pkl'
    else:
        suffix = 'SMPLX_FEMALE.pkl'

    suffix     = 'SMPLX_MALE.pkl'
    smplx_path = smplx_path + suffix

    with open(smplx_path, 'rb') as smplx_file:
        model_data = pickle.load(smplx_file, encoding='latin1')
    '''
    
    data_struct = Struct(**model_data)
    j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=torch.float32)
    joints      = vertices2joints(j_regressor, vertices) 
    
    return joints.numpy().reshape(-1,3)

def getPointCloudFeature(pointNet, cfg, obj_name, hbox, obox, imgPath, key, idx, model_data, smplx_result, vertex_choice, obj_para_dict, smplx_path=None):
    '''
        steps for generating pc
        step.1 [generate_3D_obj_Neg.py] -> otri, obj_vertexs
        step.2 [rotate_sampling_Neg.py] -> pick_vertex
        step.3 [Feature_extraction.py]  -> feature
    '''
    if smplx_path == None:
        smplx_path = cfg.SMPLX_PATH
        mesh = os.path.join(smplx_path, 'meshes/HICO_train2015_%08d/%03d.obj' % (key, idx))
    else:
        mesh = os.path.join(smplx_path, 'meshes/HICO_test2015_%08d/%03d.obj' % (key, idx))

    htri         = trimesh.load_mesh(mesh)
    vertice      = np.array(htri.vertices,dtype=np.float32)
    joints       = get_joints(model_data=model_data, \
                              vertices=torch.FloatTensor(torch.from_numpy(vertice.reshape(1,-1,3))))
    ansp         = rotate(joints - joints[0])
    shoulder_len = np.linalg.norm(joints[16] - joints[17])
    radius       = obj_para_dict[obj_name]['ratio'] * shoulder_len
    gamma_min    = obj_para_dict[obj_name]['gamma_min']
    gamma_max    = obj_para_dict[obj_name]['gamma_max']
    
    # step 1, get obj_vertexs
    otri, obj_vertexs = get_param(smplx_result, hbox, obox, htri, imgPath, radius, gamma_min, gamma_max)

    # step 2, get pick_vertex
    vertice     = vertice[vertex_choice,:]
    pick_vertex = np.vstack((vertice, obj_vertexs))
    pick_vertex = pick_vertex - joints[0]
    pick_vertex = rotate_mul(pick_vertex, ansp) # (1228, 3)
    pick_vertex = np.array(pick_vertex)
    
    # step 3, get pointNet feature
    pick_vertex = torch.from_numpy(pick_vertex.astype(np.float32)) # (1228, 3)
    pick_vertex = pick_vertex.unsqueeze(0).unsqueeze(0).cuda()     # (1, 1, 1228, 3) 
    pc          = pointNet(pick_vertex).unsqueeze(0) # 1, 1228, 256
    pc          = pc.cpu().detach().numpy()

    return pc

def get_param(result, hbox, obox, htri, img, radius=None, gamma_min=None, gamma_max=None):    
    focal_length = 5000
    root1  = pickle.load(open('./hakeaction/datasets/utils/hico_files/equation-root1.pkl',  'rb'), encoding='latin1')
    root1r = pickle.load(open('./hakeaction/datasets/utils/hico_files/equation-root1r.pkl', 'rb'), encoding='latin1')
    rotation = result['camera_rotation'][0, :, :]
    camera_transl = result['camera_translation']
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = rotation
    camera_transform[:3, 3]  = camera_transl
    camera_mat = np.eye(2).astype(np.float32) * focal_length

    vert = np.array(htri.vertices)

    img    = cv2.imread(img)[:, :, ::-1].astype(np.float32) / 255.
    camera_center = np.array([img.shape[1], img.shape[0]]) * 0.5
    camera_center = camera_center.astype(np.int32)
    
    hbox[0] -= camera_center[0]
    hbox[1] -= camera_center[1]
    hbox[2] -= camera_center[0]
    hbox[3] -= camera_center[1]
    obox[0] -= camera_center[0]
    obox[1] -= camera_center[1]
    obox[2] -= camera_center[0]
    obox[3] -= camera_center[1]

    x_mid = (obox[0] + obox[2]) / 2
    y1, y2 = obox[1], obox[3]

    t1, t2, t3 = camera_transl[0, 0], camera_transl[0, 1], camera_transl[0, 2]

    ly1_x = [x_mid / focal_length, x_mid * t3 / focal_length - t1]
    ly1_y = [-y1 / focal_length, -y1 * t3 / focal_length + t2]
    ly2_x = [x_mid / focal_length, x_mid * t3 / focal_length - t1]
    ly2_y = [-y2 / focal_length, -y2 * t3 / focal_length + t2]
    vec_1 = np.array([ly1_x[0], ly1_y[0], 1])
    vec_2 = np.array([ly2_x[0], ly2_y[0], 1])
    top = np.sum(vec_1 * vec_2)
    bottom = np.sqrt(np.sum(vec_1 * vec_1)) * np.sqrt(np.sum(vec_2 * vec_2))
    theta = np.arccos(top / bottom)

    _t1 = t1
    _t2 = t2
    _t3 = t3
    _x_mid = x_mid
    _theta = theta
    _focal_length = focal_length
    x = sympy.Symbol('x', real=True)
    y = sympy.Symbol('y', real=True)
    z = sympy.Symbol('z', real=True)
    t1 = sympy.Symbol('t1', real=True)
    t2 = sympy.Symbol('t2', real=True)
    t3 = sympy.Symbol('t3', real=True)
    x_mid = sympy.Symbol('x_mid', real=True)
    theta = sympy.Symbol('theta', real=True)
    focal_length = sympy.Symbol('focal_length', real=True)
    vec_20 = sympy.Symbol('vec_20', real=True)
    vec_21 = sympy.Symbol('vec_21', real=True)
    vec_22 = sympy.Symbol('vec_22', real=True)
    r = sympy.Symbol('r', real=True)

    maxz = np.max(vert[:, 2]) * gamma_max
    minz = np.min(vert[:, 2]) * gamma_min
    
    
    if radius is not None:
        value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                 vec_21: vec_2[1], vec_22: vec_2[2], r: radius}
        for i in range(4):
            ansx = root1[i][0].evalf(subs=value)
            ansy = root1[i][1].evalf(subs=value)
            ansz = root1[i][2].evalf(subs=value)
            y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
            x2D = (-ansx + _t1) / (ansz + _t3) * _focal_length

            if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                idx = i
    
        ansx = root1[idx][0].evalf(subs=value)
        ansy = root1[idx][1].evalf(subs=value)
        ansz = root1[idx][2].evalf(subs=value)
        
        if (ansz > maxz or ansz < minz):          
            if (ansz > maxz): ansz = maxz
            if (ansz < minz): ansz = minz
            value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                 vec_21: vec_2[1], vec_22: vec_2[2], z: ansz}
            for i in range(2):
                ansx = root1r[i][0].evalf(subs=value)
                ansy = root1r[i][1].evalf(subs=value)
                y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
                x2D = (ansx + _t1) / (ansz + _t3) * _focal_length
                if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                    idx = i
            ansx = root1r[idx][0].evalf(subs=value)
            ansy = root1r[idx][1].evalf(subs=value)
            radius = root1r[idx][2].evalf(subs=value)

        point = [float(ansx), float(ansy), float(ansz)]
        point = np.append(point, 1)
        ansr = radius
    else:
        R = cal_r_rule(vert[9448] - vert[9929], 1)
        left = R / 10
        right = R * 100
        flag, ansr, idx, flag2, flag3, tot = 0, 0, -1, 0, 0, 0
        while (flag == 0 and tot < 15):
            R = (left + right) / 2
            tot = tot + 1
            value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                     vec_21: vec_2[1], vec_22: vec_2[2], r: R}
            if (flag2 == 0):
                flag2 = 1
                for i in range(4):
                    ansx = root1[i][0].evalf(subs=value)
                    ansy = root1[i][1].evalf(subs=value)
                    ansz = root1[i][2].evalf(subs=value)
                    y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
                    x2D = (ansx + _t1) / (ansz + _t3) * _focal_length
                    if (math.isnan(y2D)):
                        flag3 = 1
                        break
                    if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                        idx = i
            if (flag3 == 1):
                break
            ansx = root1[idx][0].evalf(subs=value)
            ansy = root1[idx][1].evalf(subs=value)
            ansz = root1[idx][2].evalf(subs=value)
    
            point = [float(ansx), float(ansy), float(ansz)]
            point = np.append(point, 1)
    
            if (point[2] < minz):
                left = R
            elif (point[2] > maxz):
                right = R
            elif (point[2] >= minz and point[2] <= maxz):
                flag = 1
                ansr = float(R)
    
    # print(ansx,ansy,ansz, ansr)
    verts, faces = icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    for i in range(len(verts)):
        verts[i][0] *= ansr
        verts[i][1] *= ansr
        verts[i][2] *= ansr
        verts[i][0] += point[0]
        verts[i][1] += point[1]
        verts[i][2] += point[2]
    otri = trimesh.Trimesh(vertices=verts, faces=faces)
    hbox[0] += camera_center[0]
    hbox[1] += camera_center[1]
    hbox[2] += camera_center[0]
    hbox[3] += camera_center[1]
    obox[0] += camera_center[0]
    obox[1] += camera_center[1]
    obox[2] += camera_center[0]
    obox[3] += camera_center[1]
    return otri, verts

def Generate_action_object(idx, num_obj):
    action_obj = np.zeros([1, num_obj], dtype=np.float64)
    if isinstance(idx, int) or isinstance(idx, np.int32):
        action_obj[:, idx-1] = 1
    else:
        idx = np.array(list(idx))
        idx = idx - 1
        action_obj[:, list(idx)] = 1
    return action_obj

def load_hdf5(infile, keys):
	"""
	Load hdf5 file to dict with arrays.
	Args:
	  infile: str, name of hdf5 file
	  keys: tuple/list, keys in hdf5 dataset  
	Return:
	  dict, <key, value> in hdf5 file
	"""
	with h5py.File(infile, 'r') as f:
		return {key : f[key][:] for key in keys}

def write_hdf5(outfile, arr_dict):
	"""
	Write arrays to hdf5 file. Create outfile if not exists. 
	torch.Tensor will automatically reformat to numpy.array.
	Args:
	  outfile: str, name of hdf5 file
	  arr_dict: dict, group of data-to-write
	"""
	with h5py.File(outfile, 'w') as f:
		for key in arr_dict.keys():
			f.create_dataset(key, data=arr_dict[key])
			
def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d

def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    
    x2 = bb2[1] - bb2[0]
    y2 = bb2[3] - bb2[2]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    
    
    xiou = min(bb1[2], bb2[1]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[2])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)

def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def calc_ap(scores, bboxes, keys, hoi_id, begin, cfg):
    if keys.shape[0] == 0:
        return 0, 0
    score = scores[:, hoi_id - begin]
    hit = []
    idx = np.argsort(score)[::-1]
    # gt_bbox = pickle.load(open(cfg.DATA_DIR + '/gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'), encoding='latin1')
    gt_bbox = pickle.load(open('/Disk3/yonglu/pasta_torch/' + '/gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'), encoding='latin1')
    
    npos = 0
    used = {}
    
    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key  = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    
    return ap, np.max(rec)

def get_map(keys, scores, bboxes, cfg):
    mAP  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1]
        score = scores[i]
        bbox  = bboxes[i]
        key   = keys[i]
        for hoi_id in range(begin, end):
            mAP[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin, cfg)
    return mAP, mrec

def get_map_with_NIS(keys, scores, bboxes, pos, neg, NISThreshold=0.9):
    mAP  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1]
        score = scores[i]
        bbox  = bboxes[i]
        key   = keys[i]
        pos_i = pos[i]

        # proceed NIS (Non-Interaction Surpress)  
        NIS_inds = np.where(pos_i > NISThreshold)
        score    = score[NIS_inds]
        bbox     = bbox[NIS_inds]
        key      = key[NIS_inds]

        # calculate mAP and mREC
        for hoi_id in range(begin, end):
            mAP[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin)

    return mAP, mrec

