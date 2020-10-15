import os
import os.path as osp
import numpy as np
import pickle
import trimesh
import cv2
import matplotlib.pyplot as plt
import sympy, math
import pyrr
import torch

'''
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
    
    f = open('hico_list_hoi.txt','r')
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
    f = open('hico_obj_parameter.txt','r')
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
'''

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def vertices2joints(J_regressor, vertices):
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def get_joints(args,vertices):
    if (args.gender == 'neutral'):
        suffix = 'SMPLX_NEUTRAL.pkl'
    elif (args.gender == 'male'):
        suffix = 'SMPLX_MALE.pkl'
    else:
        suffix = 'SMPLX_FEMALE.pkl'
    smplx_path = args.smplx_path + suffix

    with open(smplx_path, 'rb') as smplx_file:
        model_data = pickle.load(smplx_file, encoding='latin1')
    
    data_struct = Struct(**model_data)
    j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=torch.float32)
    joints = vertices2joints(j_regressor, vertices) 
    return joints.numpy().reshape(-1,3)

def point_align_vis(result, obox, mesh, img):
    img = cv2.imread(img)[:, :, ::-1].astype(np.float32) / 255.   
    rotation = result['camera_rotation'][0, :, :]
    camera_trans = result['camera_translation']
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = rotation
    camera_transform[:3, 3]  = camera_trans
    camera_mat = np.zeros((2, 2))
    camera_mat[0, 0] = 5000.
    camera_mat[1, 1] = 5000
    vert = []
    with open(mesh) as f:
        while True:
            line = f.readline().split()
            if line[0] == 'v':
                vert.append(np.array([float(line[1]), float(line[2]), float(line[3])]))
            else:
                break
    vert = np.array(vert)
    camera_center = np.array([img.shape[1], img.shape[0]]) * 0.5
    camera_center = camera_center.astype(np.int32)
    homog_coord = np.ones(list(vert.shape[:-1]) + [1])
    points_h = np.concatenate([vert, homog_coord], axis=-1)
    for i in range(points_h.shape[0]):
        point = points_h[i]
        point[1] *= -1
        projected = np.matmul(camera_transform, point)
        img_point = projected[:2] / projected[2]
        img_point = np.matmul(camera_mat, img_point)
        img_point = img_point + camera_center
        img_point = img_point.astype(np.int32)
        img = cv2.circle(img, (img_point[0], img_point[1]), 5, (0, 1, 0), -1)
    img = cv2.rectangle(img, (obox[0], obox[1]), (obox[2], obox[3]),(1, 0, 0), 2)
    plt.imshow(img)

def icosahedron():
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1),
        (11, 7, 6), (11, 8, 7), (11, 9, 8), (11, 10, 9), (11, 6, 10),
        (1, 6, 2), (2, 7, 3), (3, 8, 4), (4, 9, 5), (5, 10, 1),
        (6, 7, 2), (7, 8, 3), (8, 9, 4), (9, 10, 5), (10, 6, 1),
    ]
    verts = [
        [0.000, 0.000, 1.000],  [0.894, 0.000, 0.447], [0.276, 0.851, 0.447],
        [-0.724, 0.526, 0.447], [-0.724, -0.526, 0.447], [0.276, -0.851, 0.447],
        [0.724, 0.526, -0.447], [-0.276, 0.851, -0.447], [-0.894, 0.000, -0.447],
        [-0.276, -0.851, -0.447], [0.724, -0.526, -0.447], [0.000, 0.000, -1.000],
    ]
    return verts, faces
def subdivide(verts, faces):
    triangles = len(faces)
    for faceIndex in range(triangles):

        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        a, b, c = np.float32([verts[vertIndex] for vertIndex in face])
        verts.append(pyrr.vector.normalize(a + b))
        verts.append(pyrr.vector.normalize(b + c))
        verts.append(pyrr.vector.normalize(a + c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i + 1, i + 2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])
    return verts, faces

def cal_r_rule(d, r_ratio):
    dis = np.sqrt(np.sum(d * d))
    r = dis * r_ratio
    return r
def get_param(result, hbox, obox, htri, img, radius=None, gamma_min=None, gamma_max=None):    
    focal_length = 5000
    root1  = pickle.load(open('equation-root1.pkl', 'rb'), encoding='latin1')
    root1r = pickle.load(open('equation-root1r.pkl', 'rb'), encoding='latin1')
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

def rotate_mul(verts, rotate):
    rot = np.insert(verts, 3, values = 1, axis = 1)
    ret = np.dot(rot, rotate)
    return ret[:,:3]

def rotate(joints):
    s = np.array([0., 1., 0.])
    l = np.sqrt(np.sum(s * s))
    x = s[0] / l
    y = s[1] / l
    z = s[2] / l
    
    a = 0
    b = 0
    c = 0

    u = x
    v = y
    w = z
    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w
    au = a * u
    av = a * v
    aw = a * w
    bu = b * u
    bv = b * v
    bw = b * w
    cu = c * u
    cv = c * v
    cw = c * w

    ansp = np.zeros((4,4))
    ans = 1000

    for i in range(1,1800):
      pi = math.acos(-1)
      ang = pi / 1800 * i
 
      v1 = joints[16]
      v2 = joints[17]
      
      sinA = math.sin(ang)
      cosA = math.cos(ang)
      costheta = cosA
      sintheta = sinA
      p = np.zeros((4,4))
      p[0][0] = uu + (vv + ww) * costheta
      p[0][1] = uv * (1 - costheta) + w * sintheta
      p[0][2] = uw * (1 - costheta) - v * sintheta
      p[0][3] = 0

      p[1][0] = uv * (1 - costheta) - w * sintheta
      p[1][1] = vv + (uu + ww) * costheta
      p[1][2] = vw * (1 - costheta) + u * sintheta
      p[1][3] = 0

      p[2][0] = uw * (1 - costheta) + v * sintheta
      p[2][1] = vw * (1 - costheta) - u * sintheta
      p[2][2] = ww + (uu + vv) * costheta
      p[2][3] = 0

      p[3][0] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta
      p[3][1] = (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta
      p[3][2] = (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta
      p[3][3] = 1

      v1 = v1.reshape(1,3)
      v2 = v2.reshape(1,3)
      rotv1 = np.dot(np.insert(v1, 3, values=1, axis=1),p)
      rotv2 = np.dot(np.insert(v2, 3, values=1, axis=1),p)

      if (abs(rotv1[0][2] - rotv2[0][2]) < ans):
        ans = abs(rotv1[0][2] - rotv2[0][2])
        ansp = p

    return ansp
