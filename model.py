import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import math
from collections import OrderedDict

from util.obj_80_768_averg_matrix import obj_matrix
from util.matrix_sentence_76 import sentence_only
from util.matrix_hoi import m as hoi_matrix_2304
from util.matrix_vcoco import m as hoi_matrix_vcoco
from util.matrix_sentence import m_sentence as sentence_pvp93
from util.matrix_ava import m as ava_matrix
from util.AVA_utils import action

class BNView(nn.Module):
    def __init__(self, n_feat):
        super(BNView, self).__init__()
        self.n_feat = n_feat
        self.bn = nn.BatchNorm1d(n_feat)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], -1, self.n_feat)
        x = x.transpose(1, -1)
        x = self.bn(x)
        x = x.transpose(1, -1)
        x = x.reshape(x_shape)
        return x


def three_layer_mlp(
        input_dim, hidden_dim, output_dim, act="relu", dropout=0.0, bn=False
):
    act_dict = {"relu": nn.ReLU, "tanh": nn.Tanh}
    layers = [
        ("fc1", nn.Linear(input_dim, hidden_dim)),
        ("act1", act_dict[act]()),
    ]
    assert dropout == 0.0 or not bn
    if dropout > 0:
        layers.append(("dropout1", nn.Dropout(dropout, inplace=True)))
    if bn:
        layers.append(("batchnorm1", BNView(hidden_dim)))
    layers.append(("fc2", nn.Linear(hidden_dim, output_dim)))
    return nn.Sequential(OrderedDict(layers))


class MHAtt(nn.Module):
    def __init__(self, n_heads=4, dim=64, dropout=0.0):
        super(MHAtt, self).__init__()
        self.n_head, self.dim = n_heads, dim
        self.query, self.key, self.value = (
            nn.Linear(dim, n_heads * dim) for _ in range(3)
        )
        self.linear = nn.Linear(n_heads * dim, dim)
        self.att_norm = nn.LayerNorm(dim)
        self.ffn = three_layer_mlp(dim, 4 * dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (bz,18,10,64)
        prev_shape = x.shape[:-2]  # (bz,18)
        query = self.query(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,4,64)
        key = self.key(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,4,64)
        value = self.value(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,4,64)

        query, key, value = (
            query.transpose(-2, -3),
            key.transpose(-2, -3),
            value.transpose(-2, -3),
        ) # (bz,18,4,10,64)

        att = F.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / int(np.sqrt(self.dim)), dim=-1
        ) # (bz,18,4,10,64)*(bz,18,4,64,10) =  (bz,18,4,10,10) -> (bz,18,4,10,10)
        linear = self.linear(
            torch.matmul(att, value)
                .transpose(-2, -3)
                .reshape(*prev_shape, -1, self.n_head * self.dim)
        ) # (bz,18,4,10,10)*(bz,18,4,10,64)= (bz,18,4,10,64)->(bz,18,10,4,64)->(bz,18,10,4*64)->(bz,18,10,64)
        att_norm = self.att_norm(x + linear)

        ffn = self.ffn_norm(att_norm + self.ffn(att_norm))

        return ffn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


acti = {
    'ReLU': nn.ReLU,
    'Swish': Swish,
}


class NLR_simplified_no_T(nn.Module):
    def __init__(self, config):
        super(NLR_simplified_no_T, self).__init__()

        self.config = config.MODEL
        self.is_training = True

        if self.config.NUM_PVP == 93:
            self.pvp_matrix = np.array(sentence_pvp93, dtype="float32")
            self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (93,2304)
        elif self.config.NUM_PVP == 76:
            if self.config.NUM_CLASS in [80, 60]:
                self.pvp_matrix = np.array(sentence_only, dtype="float32")
            else:
                self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
            self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
            self.obj_matrix = np.array(obj_matrix, dtype="float32")
            self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)
        else:
            raise NotImplementedError

        if self.config.NUM_CLASS == 600:
            self.hoi_matrix = np.array(hoi_matrix_2304, dtype="float32")
        elif self.config.NUM_CLASS == 29:
            self.hoi_matrix = np.array(hoi_matrix_vcoco, dtype="float32")[:, :1536]
        elif self.config.NUM_CLASS == 80:
            self.hoi_matrix = np.array(ava_matrix, dtype="float32")
        elif self.config.NUM_CLASS == 60:
            self.hoi_matrix = np.array(ava_matrix, dtype="float32")
            self.hoi_matrix = self.hoi_matrix[np.array(action) - 1, ...]
        else:
            raise NotImplementedError
        self.hoi_matrix = torch.from_numpy(self.hoi_matrix).cuda()  # (600,1536)

        self.pvp_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
        )

        self.hoi_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
        )

        self.not_layers = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
        )

        self.judger = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            1,
        )
        self.judger_loss = nn.BCEWithLogitsLoss()

        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.or_attn = MHAtt(dim=self.config.EMBEDDING_DIM)

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCELoss()

        if not self.config.get('DYNAMIC', False):
            self.mh_att = MHAtt(dim=self.config.EMBEDDING_DIM)
            self.vote_final = three_layer_mlp(
                self.config.NUM_RULE * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
            )

        if self.config.get("COS_PROB", False):
            self.pvp_prob_embedding = three_layer_mlp(
                self.config.EMBEDDING_DIM * 2,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM
            )

        if self.config.get('JUDGER_COMB') and self.config.JUDGER_COMB:
            self.judger_weight = torch.nn.Parameter(
                torch.randn([self.config.NUM_CLASS, self.config.NUM_RULE]), requires_grad=True
            )

        self.load_pretrain()

    def load_pretrain(self):
        if not self.config.get('CHECKPOINT') or not self.config.CHECKPOINT:
            return
        pretrained_dict = torch.load(self.config.CHECKPOINT)['state'].state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        if self.config.get('RULE_ADJUST') and self.config.RULE_ADJUST:
            for k, v in self.named_parameters():
                if 'vote_final' in k or 'mh_att' in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        print('pretrained loaded.')

    def set_status(self, is_training=True):
        self.is_training = is_training

    def not_module(self, input):
        return self.not_layers(input)

    def or_module(self, input, lens=None, key="share"):
        # input (...,x,embed_dim), lens (-1,)
        # or operation on 'x' vectors
        # Take the first as output, cat every vector with output, conduct do as new output
        # This is essentially RNN
        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            if input.shape[-1] == self.config.EMBEDDING_DIM:
                return self.or_attn(input).mean(-2)
            else:
                return self.or_attn(
                    input.reshape(*input.shape[:-1], -1, self.config.EMBEDDING_DIM)
                ).mean(-2)

        output = input[..., 0, :]
        for i in range(1, input.shape[-2]):
            output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))
        return output

    def logic_loss_x_or_x(self, x):
        # x or x == x
        reg = self.judger_loss(
            self.judger(self.or_layers(torch.cat([x, x], dim=-1))),
            self.judger(x).sigmoid()
        )
        return reg

    def logic_loss_x_or_not_x(self, x):
        # x or (not x) == T
        # T(embed_dim) -> T(x.shape),
        reg = self.judger_loss(self.judger(self.or_layers(torch.cat([x, self.not_module(x)], dim=-1))).squeeze(),
                               x.new_ones(x.shape[:-1]))
        return reg

    def logic_loss(self, batch):
        # logical_regularizer, not
        # not+T == F
        # (...,embed_dim)->(-1,embed_dim)
        reg_1 = self.judger_loss(self.judger(self.not_output), 1 - self.judger(self.pos_input).sigmoid())
        # not+not+x == x
        not_not_output = self.not_module(self.not_output)
        reg_2 = self.judger_loss(self.judger(not_not_output), self.judger(self.pos_input).sigmoid())
        # logical_regularizer, or
        # x or T == T
        # not_output(bz,18,10,k,embed_dim), or_77_output(bz,18,10,embed_dim), hoi_L(bz,18,10,embed_dim)
        or_reg_targets = [self.not_output, self.or_77_output, self.hoi_L]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

        # print([reg_target.shape for reg_target in or_reg_targets])
        # reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in or_reg_targets)
        # reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in or_reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in or_reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in or_reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def prob_loss(self, batch):
        return ((1 - self.gt_pvp) * F.cosine_similarity(self.pvp_L, self.pvp_L_embed, dim=-1) -
                self.gt_pvp * F.cosine_similarity(self.pvp_L, self.not_pvp_L_embed, dim=-1)).abs().mean()

    def forward(self, batch):
        # batch['rule'] : (bz,18,10,k)
        # 18 rule selected for each pair(according to gt_obj); k/76 pos where rule=1, index, currently k=22
        # batch['gt_label']: (bz,18)
        # batch['gt_range']: (bz,18)
        self.or_layers = self.or_module
        batch_size = batch["rule"].shape[0]
        self.config.NUM_RULE = batch["rule"].shape[-2]

        if "gt_pvp" in batch.keys():
            self.gt_pvp = batch["gt_pvp"]
        else:
            self.gt_pvp = torch.cat([batch[key] for key in self.config.L_keys[:-1]], dim=-1)

        if self.config.get("ADD_NON", False):
            assert not self.config.get("COS_PROB", False)
            self.pvp_L = self.pvp_matrix.expand(batch_size, -1, -1)
            if not self.config.NUM_CLASS in [80, 60]:
                self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
                self.pvp_L = torch.cat(
                    [
                        self.pvp_L,
                        self.obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1),
                    ],
                    dim=2,
                )  # (bz,76,2304)

            # (bz,76,embed_dim)
            self.pvp_L = self.pvp_embedding(self.pvp_L)
            self.pvp_L = self.gt_pvp[..., None] * self.pvp_L + (
                    1 - self.gt_pvp[..., None]
            ) * self.not_module(self.pvp_L)
        elif self.config.get("COS_PROB", False):
            self.pvp_L = self.pvp_matrix.expand(batch_size, -1, -1)
            self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
            self.pvp_L = torch.cat(
                [
                    self.pvp_L,
                    self.obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1),
                ],
                dim=2,
            )  # (bz,76,2304)

            # (bz,76,embed_dim)
            self.pvp_L_embed = self.gt_pvp[..., None] * self.pvp_embedding(self.pvp_L)
            self.not_pvp_L_embed = (1 - self.gt_pvp[..., None]) * self.not_module(self.pvp_L_embed)

            self.pvp_L = self.pvp_prob_embedding(torch.cat((
                self.pvp_L_embed, self.not_pvp_L_embed), dim=-1))
        else:
            # (bz,76),(76,1576)->(bz,76,1576)
            # print(self.gt_pvp.shape, self.pvp_matrix.shape)
            self.pvp_L = torch.einsum("ij,jk->ijk", self.gt_pvp, self.pvp_matrix)
            if self.config.NUM_CLASS not in [80, 60]:
                # (bz,80)*(80,768)=(bz,768)
                self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
                self.pvp_L = torch.cat(
                    [
                        self.pvp_L,
                        self.obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1),
                    ],
                    dim=2,
                )  # (bz,76,2304)
            # (bz,76,2304)->(bz,76,embed_dim)
            self.pvp_L = self.pvp_embedding(self.pvp_L)

        # input rule is index where label=1
        # batch_idx: (bz,)->(bz,1,1,1)->(bz,18,10,k)
        batch_idx = (torch.arange(batch_size).view(-1, 1, 1, 1).expand(batch["rule"].shape))

        # batch_idx: (bz,18,10,k), batch['rule']: (bz,18,10,k), pvp_L: (bz,76,embed_dim)
        # pos_input: (bz,18,10,k,embed_dim)
        self.pos_input = self.pvp_L[batch_idx, batch["rule"], :]
        self.not_output = self.not_module(self.pos_input)
        # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
        self.or_76_output = self.or_module(self.not_output)

        if self.config.NUM_CLASS == 600:
            self.hoi_L = self.hoi_matrix[batch["gt_range"], :]
        elif self.config.NUM_CLASS == 29:
            self.hoi_L = self.hoi_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (29,1536) -> (bz,29,1536)
            self.hoi_L = torch.cat(
                [
                    self.hoi_L,
                    self.obj_L.unsqueeze(1).expand(
                        *self.hoi_L.shape[:-1], -1
                    ),
                ],
                dim=-1,
            )  # (bz,18,2304)
        elif self.config.NUM_CLASS in [80, 60]:
            self.hoi_L = self.hoi_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (bz,80,2304)
        else:
            raise NotImplementedError

        # (bz,76,2304)->(bz,76,embed_dim)
        self.hoi_L = self.hoi_embedding(self.hoi_L)
        self.hoi_L = self.hoi_L.unsqueeze(2).expand(
            -1, -1, self.config.NUM_RULE, -1
        )  # (bz,18,10,embed_dim)

        # (bz,18,10,2*embed_dim)->(or layer)->(bz,18,10,embed_dim)
        self.or_77_output = self.or_layers(
            torch.cat([self.hoi_L, self.or_76_output], dim=-1)
        )


        output = {}
        if self.config.get('JUDGER_COMB') and self.config.JUDGER_COMB:

            vote = self.judger(self.or_77_output).squeeze()  # (bz,18,10)
            # self.s = torch.sum(vote * self.judger_weight[batch["gt_range"], :],
            #                    dim=-1)  # (bz,18)*(bz,18,10) -> (bz,18,10) -> (bz,18)
            # (bz,18,10)*(bz,18,10) -> (bz,18)
            self.s = torch.einsum("ijk,ijl->ij", vote, self.judger_weight[batch["gt_range"], :].softmax(-1))

            if self.config.get('UPDATE') and self.config.UPDATE:
                for i in range(self.config.NUM_RULE):
                    output["s" + str(i)] = vote[..., i]
                    output["p" + str(i)] = self.probability(output["s" + str(i)])

                    if self.config.get('CAL_SEPARATE_LOSS') and self.config.CAL_SEPARATE_LOSS:
                        output["L_cls" + str(i)] = batch["gt_label"] * torch.log(output["p" + str(i)]) + (
                                1 - batch["gt_label"]) * torch.log(1 - output["p" + str(i)])  # (bz,18)

                return output

        elif self.config.get('JUDGER_AVG') and self.config.JUDGER_AVG:

            vote = self.judger(self.or_77_output).squeeze()  # (bz,18,10)
            if not self.config.get('DYNAMIC', False):
                self.s = vote
                if len(vote.shape)>2:
                    self.s = torch.mean(vote, dim=-1)  # (bz,18)
                
            else:
                # batch["rule_cnt"], (bz, 18)
                extend_shape = [*batch["rule_cnt"].shape, self.config.NUM_RULE]
                rule_cnt = batch["rule_cnt"].unsqueeze(-1).expand(extend_shape)  # (bz,18) -> (bz,18,10)
                mask = torch.arange(self.config.NUM_RULE).view(1, 1, -1).expand(extend_shape).cuda()
                mask = torch.where(mask < rule_cnt, torch.ones_like(mask), torch.zeros_like(mask))
                vote = vote.masked_fill(mask=(1 - mask).bool(), value=0)  # (bz,18,10)
                self.s = torch.sum(vote, dim=-1) / batch["rule_cnt"]

            if self.config.get('UPDATE') and self.config.UPDATE:
                for i in range(self.config.NUM_RULE):
                    output["s" + str(i)] = vote[..., i]
                    output["p" + str(i)] = self.probability(output["s" + str(i)])  # (bz,18)
                    if "gt_label" in batch.keys():
                        output["L_cls" + str(i)] = batch["gt_label"] * torch.log(output["p" + str(i)]) + (
                                1 - batch["gt_label"]) * torch.log(1 - output["p" + str(i)])  # (bz,18)

                return output

        else:
            # or_77_output: (bz,18,10,embed_dim)
            mh_att = self.mh_att(self.or_77_output)  # (bz,18,10,embed_dim)

            if self.config.get('SEPARATE_EVAL', False):
                for i in range(self.config.NUM_RULE):
                    output["s" + str(i)] = self.judger(mh_att[..., i, :]).squeeze()  # (bz,18,embed_dim)->(bz,18)
                    output["p" + str(i)] = self.probability(output["s" + str(i)])

            # print(self.or_77_output.shape, mh_att.shape)
            if self.config.get('UPDATE') and self.config.UPDATE:
                for i in range(self.config.NUM_RULE):
                    # output["s" + str(i)] = self.judger(
                    #     self.or_77_output[..., i, :]).squeeze()  # (bz,18,embed_dim)->(bz,18,1)
                    output["s" + str(i)] = self.judger(mh_att[..., i, :]).squeeze()  # (bz,18,embed_dim)->(bz,18)
                    output["p" + str(i)] = self.probability(output["s" + str(i)])
                    output["L_cls" + str(i)] = batch["gt_label"] * torch.log(output["p" + str(i)]) + (
                            1 - batch["gt_label"]) * torch.log(1 - output["p" + str(i)])  # (bz,18)
                    # output["L_cls" + str(i)] = self.classification_loss(output["p" + str(i)], batch["gt_label"])

                return output

            self.or_vote_output = self.vote_final(
                mh_att.reshape(*self.or_77_output.shape[:2], -1)
            )  # (bz,18,10,embed_dim)->(bz,18,10*embed_dim)->(vote_final)->(bz,18,embed_dim)

            self.s = self.judger(self.or_vote_output).squeeze()

        self.p = self.probability(self.s)

        output["s"] = self.s
        output["p"] = self.p

        if "gt_label" in batch.keys() or "labels_v" in batch.keys() or "labels_r" in batch.keys() or "labels_a" in batch.keys():
            if "gt_label" in batch.keys():
                output["L_cls"] = self.classification_loss(self.p, batch["gt_label"])
            elif "labels_v" in batch.keys():
                output["L_cls"] = self.classification_loss(self.p, batch["labels_v"])
            elif "labels_r" in batch.keys():
                output["L_cls"] = self.classification_loss(self.p, batch["labels_r"])
            elif "labels_a" in batch.keys():
                output["L_cls"] = self.classification_loss(self.p, batch["labels_a"])

            output["L_logic"] = self.logic_loss(batch)
            output["loss"] = self.config.LOSS.CLS_FAC * output["L_cls"] + self.config.LOSS.LOGIC_FAC * output["L_logic"]

            if self.config.get("COS_PROB", False):
                output["L_prob"] = self.prob_loss(batch)
                output["loss"] += output["L_prob"]

        return output