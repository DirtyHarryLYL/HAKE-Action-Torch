import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from collections import OrderedDict
import math

verb_mapping = pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')

class TLayer(nn.Module):
    def __init__(self, num, in_features, out_features, bias=True):
        super(TLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num          = num
        self.weight       = nn.Parameter(torch.Tensor(num, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = torch.einsum('ilk,kjl->ijk', [input, self.weight]) + self.bias
        return output

    def extra_repr(self):
        return 'num={}, in_features={}, out_features={}, bias={}'.format(
            self.num, self.in_features, self.out_features, self.bias is not None
        )

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        assert config.LAYER_SIZE[0] == config.LAYER_SIZE[-1]
        self.num_classes = config.NUM_CLASSES
        self.trans = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.trans['trans%d' % i] = TLayer(config.NUM_CLASSES, config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i], 1)
            if i < len(config.LAYER_SIZE) - 1:
                self.trans['bn%d' % i]   = nn.BatchNorm1d(config.LAYER_SIZE[i])
                self.trans['relu%d' % i] = nn.ReLU()
        self.trans = nn.Sequential(self.trans)

    def forward(self, uni, cat):
        if len(uni.shape) < 3:
            uni = torch.unsqueeze(uni, 2).expand(-1, -1, self.num_classes)
        if len(cat.shape) < 3:
            cat = torch.unsqueeze(cat, 2).expand(-1, -1, self.num_classes)
        transformed = self.trans(cat)
        bias = uni - transformed
        score = torch.mean(bias * bias, dim=1) * -1
        return transformed, score

class AE(nn.Module):
    def __init__(self, config, hoi_weight):
        super(AE, self).__init__()
        
        self.config = config
        
        self.spatial = OrderedDict()
        for i in range(1, len(self.config.AE.SPATIAL.LAYER_SIZE)):
            self.spatial['fc%d' % i]   = nn.Linear(self.config.AE.SPATIAL.LAYER_SIZE[i - 1],  self.config.AE.SPATIAL.LAYER_SIZE[i])
            if self.config.AE.BN:
                self.spatial['bn%d' % i] = nn.BatchNorm1d(self.config.AE.SPATIAL.LAYER_SIZE[i])
            self.spatial['relu%d' % i] = nn.ReLU()
            if self.config.AE.DROPOUT:
                self.spatial['dropout%d' % i] = nn.Dropout(self.config.AE.DROPOUT)
        self.spatial = nn.Sequential(self.spatial)
        
        self.encoder = OrderedDict()
        for i in range(1, len(self.config.AE.ENCODER.LAYER_SIZE)):
            self.encoder['fc%d' % i]  = nn.Linear(self.config.AE.ENCODER.LAYER_SIZE[i - 1], self.config.AE.ENCODER.LAYER_SIZE[i])
            if i < len(self.config.AE.ENCODER.LAYER_SIZE) - 1:
                if self.config.AE.BN:
                    self.encoder['bn%d' % i] = nn.BatchNorm1d(self.config.AE.ENCODER.LAYER_SIZE[i])
                self.encoder['relu%d' % i] = nn.ReLU()
                if self.config.AE.DROPOUT:
                    self.encoder['dropout%d' % i] = nn.Dropout(self.config.AE.DROPOUT)
        self.encoder = nn.Sequential(self.encoder)
        
        self.decoder = OrderedDict()
        for i in reversed(range(1, len(self.config.AE.ENCODER.LAYER_SIZE))):
            self.decoder['fc%d' % (len(self.config.AE.ENCODER.LAYER_SIZE) - i)]   = nn.Linear(self.config.AE.ENCODER.LAYER_SIZE[i], self.config.AE.ENCODER.LAYER_SIZE[i - 1])
            if i > 1:
                if self.config.AE.BN:
                    self.decoder['bn%d' % (len(self.config.AE.ENCODER.LAYER_SIZE) - i)] = nn.BatchNorm1d(self.config.AE.ENCODER.LAYER_SIZE[i - 1])
                self.decoder['relu%d' % (len(self.config.AE.ENCODER.LAYER_SIZE) - i)] = nn.ReLU()
                if self.config.AE.DROPOUT:
                    self.decoder['dropout%d' % (len(self.config.AE.ENCODER.LAYER_SIZE) - i)] = nn.Dropout(self.config.AE.DROPOUT)
        self.decoder = nn.Sequential(self.decoder)
        
        self.classifier  = nn.Linear(self.config.AE.ENCODER.LAYER_SIZE[-1], self.config.AE.NUM_CLASSES)
        self.probability = nn.Sigmoid()
        
        self.reconstruction_loss = nn.MSELoss()
        
        if config.AE.BIN:
            self.binary_fac = config.AE.BIN_FAC 
            
            if config.AE.BIN_LOSS_MODE == 1 : # two labels,BCE loss
                self.binary_classifier = nn.Linear(config.AE.ENCODER.LAYER_SIZE[-1], 2)
                if self.config.AE.BIN_WEIGHT:
                    binary_weight = np.array([config.TRAIN.DATASET.NUM_NEG, 1])
                    binary_weight = torch.from_numpy(binary_weight)
                    self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=binary_weight)
                else:
                    self.binary_loss = nn.BCEWithLogitsLoss()
            elif config.AE.BIN_LOSS_MODE == 2:
                self.binary_classifier = nn.Linear(config.AE.ENCODER.LAYER_SIZE[-1], 1)
                self.binary_loss = nn.BCEWithLogitsLoss()
            else:
                self.binary_classifier = nn.Linear(config.AE.ENCODER.LAYER_SIZE[-1], 2)
                self.binary_loss = nn.CrossEntropyLoss()

        if self.config.AE.NUM_CLASSES == 600:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)  
            self.key = 'labels_sro'
        elif self.config.AE.NUM_CLASSES == 29:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)  
            self.key = 'labels'
        else:
            verb_weight  = np.matmul(verb_mapping, hoi_weight.transpose(1, 0).numpy())
            verb_weight  = torch.from_numpy((verb_weight.reshape(1, -1) / np.sum(verb_mapping, axis=1).reshape(1, -1)))
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=verb_weight)
            self.key = 'labels_r'
        
        self.reconstruction_fac  = self.config.AE.LOSS.RECONSTRUCTION_FAC
        self.classification_fac  = self.config.AE.LOSS.CLASSIFICATION_FAC
        
        if config.AE.CHECKPOINT:
            state = torch.load(config.AE.CHECKPOINT)['state']
            self.load_state_dict(state)

    def forward(self, batch):
        sp = self.spatial(batch['spatial'])
        n  = sp.shape[0]
        x  = torch.cat([batch['uni_vec'], sp], dim=1)
        z  = self.encoder(x)
        x_ = self.decoder(z)
        s  = self.classifier(z)
        p  = self.probability(s)
        
        L_cls = self.classification_fac * self.classification_loss(s, batch[self.key])
        L_rec = self.reconstruction_fac * self.reconstruction_loss(x, x_)
        
        output = {}
        output['z'] = z
        output['s'] = s
        output['p'] = p
        output['L_cls'] = L_cls
        output['L_rec'] = L_rec
        output['loss'] = L_cls + L_rec
        
        if self.config.AE.BIN:
            s_bin = self.binary_classifier(z)
            neg_label =  1 - batch['labels_r'][:, 0:1]
            pos_label =  1 - neg_label # 写反了
            
            if self.config.AE.BIN_LOSS_MODE == 1:
                binary_label = torch.cat([self.pos_label,neg_label],1)
            elif self.config.AE.BIN_LOSS_MODE == 2:
                binary_label = pos_label
            else:
                neg_label = neg_label.reshape((n,1))
                binary_label = neg_label.long()

            L_bin = self.binary_fac * self.binary_loss(s_bin, binary_label)
            p_binary = self.probability(s_bin)
            output['p_binary'] = p_binary
            output['loss'] += L_bin
        
        return output

diff_layers = {
    'Transformer': Transformer
}

class IDN(nn.Module):
    def __init__(self, config, hoi_weight):
        super(IDN, self).__init__()
        self.config = config
        
        if config.IDN.PROJ:
            self.proj = nn.Linear(2048, 4096)
        
        assert self.config.IDN.LAYER_SIZE[0] == self.config.AE.ENCODER.LAYER_SIZE[-1]
        self.AE = AE(config, hoi_weight)
        
        self.diff  = diff_layers[self.config.IDN.NAME](self.config.IDN)
        
        if config.IDN.NUM_CLASSES == 600:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)  
            self.key = 'labels_sro'
        if config.IDN.NUM_CLASSES == 29:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)  
            self.key = 'labels'
        else:
            verb_weight  = np.matmul(verb_mapping, hoi_weight.transpose(1, 0).numpy())
            verb_weight  = torch.from_numpy((verb_weight.reshape(1, -1) / np.sum(verb_mapping, axis=1).reshape(1, -1)))
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=verb_weight)
            self.key = 'labels_r'

        self.classification_fac  = self.config.IDN.CLASSIFICATION_FAC
        self.autoencoder_fac     = self.config.IDN.AUTOENCODER_FAC

        if config.IDN.REVERSE:
            self.reverse = diff_layers[self.config.IDN.NAME](self.config.IDN)
            self.reverse_fac = config.IDN.REVERSE_FAC
            
        if config.IDN.BINARY:
            self.binary_fac  = self.config.IDN.BINARY_FAC

        if config.IDN.CHECKPOINT:
            state = torch.load(config.IDN.CHECKPOINT)['state']
            self.load_state_dict(state)
            

    def forward(self, batch):
    
        output_AE = self.AE(batch)
        if self.config.IDN.PROJ:
            batch['uni_vec'] = self.proj(batch['sub_vec'] + batch['obj_vec'])
        else:
            batch['uni_vec'] = torch.cat([batch['sub_vec'], batch['obj_vec']], dim=1)
        output_DE = self.AE(batch)
        
        uni = output_AE['z']
        cat = output_DE['z']
        tran, score = self.diff(uni, cat)
        
        L_cls = self.classification_fac * self.classification_loss(score, batch[self.key])
        
        output = {}
        output['s'] = score
        output['s_AE'] = output_AE['s']
        output['p'] = torch.exp(score)
        output['L_cls'] = L_cls
        output['L_ae'] = self.autoencoder_fac * output_AE['loss']
        output['loss'] = L_cls + output['L_ae']

        if self.config.IDN.REVERSE:
            rev, score_r   = self.reverse(cat, tran)
            output['L_rev']  = self.reverse_fac * self.classification_loss(score_r, batch[self.key])
            output['loss']  += output['L_rev']
            output['s_rev']  = score_r

        if self.config.IDN.BINARY:
            tran     = tran.permute(0,2,1)
            tran_bin = self.AE.binary_classifier(tran).squeeze(2)
            if (len(batch['pos_ind']) > 0):
                L_bin_tran = self.classification_loss(tran_bin[batch['pos_ind'], :], batch['labels_r'][batch['pos_ind'], :])
            else:
                L_bin_tran = 0
            L_bin_cat = torch.mean(output_DE['p_binary'])
            output['L_bin'] = self.binary_fac * (L_bin_cat + L_bin_tran)
            output['loss'] += output['L_bin']

        return output
