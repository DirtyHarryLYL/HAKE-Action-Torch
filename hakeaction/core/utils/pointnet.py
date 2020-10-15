from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

'''
    Code source:
    https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
'''

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class PointNetHico(nn.Module):
    def __init__(self, K=64, feature_transform=False):
        super(PointNetHico, self).__init__()
        '''
        in_channels: int, out_channels: int, 
        kernel_size: Union[T, Tuple[T, T]], 
        stride: Union[T, Tuple[T, T]] = 1,
        padding: Union[T, Tuple[T, T]] = 0, 
        dilation: Union[T, Tuple[T, T]] = 1, 
        groups: int = 1, 
        bias: bool = True, 
        padding_mode: str = 'zeros'
        '''
        self.K = K

        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(1,  64, kernel_size=[1,3], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU() )
        
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU() )
        
        self.tconv1 = nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU() )
            
        self.tconv2 = nn.Sequential(
            torch.nn.Conv2d(64,  128,  kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(128),
            nn.ReLU() )

        self.tconv3 = nn.Sequential(
            torch.nn.Conv2d(128, 1024, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(1024),
            nn.ReLU() )

        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=[1228, 1], stride=[2, 2])

        self.tfc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU() )
            
        self.tfc2 = nn.Sequential(
            nn.Linear(512,  256),
            nn.BatchNorm1d(256),
            nn.ReLU() )
            
        self.weights = torch.zeros(256, self.K*self.K).cuda()
        self.biases  = torch.zeros(self.K*self.K).cuda()

        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU() )
            
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(128),
            nn.ReLU() )
            
        self.conv5 = nn.Sequential(
            torch.nn.Conv2d(128, 1024, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(1024),
            nn.ReLU() )
            
        self.conv6 = nn.Sequential(
            torch.nn.Conv2d(1088, 512, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(512),
            nn.ReLU() )
            
        self.conv7 = nn.Sequential(
            torch.nn.Conv2d(512,  256, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU() )
        

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # feature_transform_net
        transform = self.tconv1(conv2)
        transform = self.tconv2(transform)
        transform = self.tconv3(transform)
        transform = self.maxPool1(transform)
        transform = transform.view(1, -1) # [1,1024]

        transform   = self.tfc1(transform)
        transform   = self.tfc2(transform)
        self.biases += torch.Tensor(np.eye(64).flatten()).cuda()
        transform   = torch.matmul(transform, self.weights)
        transform   += self.biases
        transform   = transform.view(1, 64, 64)

        net_transformed = conv2.squeeze(3).permute(0,2,1).matmul(transform) # [1, 1228, 64]
        point_feat      = net_transformed.unsqueeze(3).permute(0,2,3,1) # [1, 64, 1, 1228]

        x = self.conv3(point_feat)
        x = self.conv4(x)
        x = self.conv5(x) # [1, 1024, 1, 1228]
        x = x.squeeze(2).unsqueeze(3)
        
        global_feat        = self.maxPool1(x)                    # [1, 1024, 1, 1]
        global_feat_expand = global_feat.repeat(1, 1, 1, 1228)   # [1, 1024, 1, 1228]

        # global_feat = self.maxPool1(x.squeeze(2).unsqueeze(3)) # torch.Size([1, 1024, 1, 1])
        concat_feat = torch.cat([point_feat, global_feat_expand], axis=1)

        x = self.conv6(concat_feat)
        x = self.conv7(x)
        x = x.view(1228, 256)

        return x
        # return conv1, transform, point_feat, global_feat, global_feat_expand, concat_feat, x

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
