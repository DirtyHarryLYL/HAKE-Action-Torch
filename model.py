import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import math
from collections import OrderedDict

from obj_80_768_averg_matrix import obj_matrix
from matrix_sentence_76 import sentence_only
# from matrix_hoi_1536 import m as hoi_matrix
from matrix_hoi import m as hoi_matrix_2304

from matrix_vcoco import m as hoi_matrix_vcoco

from matrix_sentence import m_sentence as sentence_pvp93
from matrix_ava import m as ava_matrix

from AVA_utils import action


# verb_mapping = pickle.load(open("Data/verb_mapping.pkl", "rb"), encoding="latin1")


class Boost_Rule(nn.Module):
    def __init__(self, config, hoi_weight):
        super(Boost_Rule, self).__init__()

        self.config = config
        self.num_rule = config.MODEL.NUM_RULE

        self.w = torch.nn.Parameter(
            torch.randn([600, self.num_rule]), requires_grad=True
        )

        self.probability = nn.Sigmoid()

        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)

        self.is_training = True

    def set_status(self, is_training=True):
        self.is_training = is_training

    def forward(self, batch):
        x = batch["vote"]

        y = x * self.w
        s = torch.sum(y, dim=2)

        p = self.probability(s)

        output = {}
        output["y"] = y
        output["s"] = s
        output["p"] = p

        if self.is_training:
            loss = self.classification_loss(s, batch["label"])
            output["loss"] = loss

        return output


class Linear(nn.Module):
    def __init__(self, config, hoi_weight):
        super(Linear, self).__init__()
        self.config = config.MODEL

        self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        self.obj_matrix = np.array(obj_matrix, dtype="float32")
        self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)

        self.linear = nn.Linear(
            self.config.NUM_PVP * self.config.BERT_DIM, self.config.NUM_CLASS
        )

    def forward(self, batch):
        # batch['gt_label']: (bz,600)

        # (bz,76),(76,1576)->(bz,1576)
        pvp_L = torch.einsum("ij,jk->ijk", batch["gt_pvp"], self.pvp_matrix)
        # (bz,80)*(80,768)=(bz,768)
        obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
        pvp_L = torch.cat(
            [pvp_L, obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1)], dim=2
        )  # (bz,76,2304)
        pvp_L = pvp_L.flatten(1, 2)  # (bz,76*2304)
        s = self.linear(pvp_L)

        p = self.probability(s)

        output = {}
        output["s"] = s
        output["p"] = p

        if "gt_label" in batch.keys():
            output["L_cls"] = self.classification_loss(s, batch["gt_label"])
            output["loss"] = output["L_cls"]

        return output


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


class NLR(nn.Module):
    def __init__(self, config, hoi_weight):
        super(NLR, self).__init__()
        self.config = config.MODEL
        self.is_training = True

        if self.config.get("UNIFORM_T", False):
            self.T = torch.rand(self.config.EMBEDDING_DIM).cuda() * 2 - 1
        else:
            self.T = torch.randn(self.config.EMBEDDING_DIM).cuda()

        self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        self.obj_matrix = np.array(obj_matrix, dtype="float32")
        self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)
        self.hoi_matrix = np.array(hoi_matrix, dtype="float32")
        self.hoi_matrix = torch.from_numpy(self.hoi_matrix).cuda()  # (600,1536)

        nons = [-1, 11, 21, 26, 57, 62, 75]  # -1 for np.diff
        self.non_matrix = self.pvp_matrix[
                          [non for non, rep in zip(nons[1:], np.diff(nons)) for _ in range(rep)], :
                          ]
        for i in range(1, len(nons)):
            self.non_matrix[nons[i], :] = self.non_matrix[
                                          (nons[i - 1] + 1): nons[i], :
                                          ].mean(0)

        # self.rule_mapping = torch.cat((torch.arange(63, 76), torch.arange(58, 63), torch.arange(27, 58),
        #                                torch.arange(22, 27), torch.arange(12, 22), torch.arange(0, 12))).cuda()

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

        if self.config.get('CALIB', False):
            print('pvp calib.')
            self.pvp_calib = three_layer_mlp(
                self.config.NUM_PVP,
                self.config.NUM_PVP,
                self.config.NUM_PVP,
            )

        self.pvp_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.hoi_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.not_layers = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
        )

        self.OR_LAYER_TYPE = "rnn"
        if self.config.get("OR_LAYER_TYPE", "rnn") == "rnn":
            self.or_layers = three_layer_mlp(
                2 * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                act=self.config.get("OR_ACT", "relu"),
                dropout=self.config.get("DROPOUT", 0.0),
            )
        elif self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.OR_LAYER_TYPE = "att"
            self.or_attn = MHAtt(dim=self.config.EMBEDDING_DIM)
        else:
            raise NotImplementedError

        self.probability = nn.Sigmoid()

        if self.config.LOSS.MODE == "CrossEntropy":
            self.classification_loss = nn.CrossEntropyLoss()
        else:
            self.classification_loss = nn.BCELoss()

        if self.config.NUM_CLASS == 117:
            verb_weight = np.matmul(verb_mapping, hoi_weight.transpose(1, 0).numpy())
            verb_weight = torch.from_numpy(
                (
                        verb_weight.reshape(1, -1)
                        / np.sum(verb_mapping, axis=1).reshape(1, -1)
                )
            ).float()
            self.classification_loss = nn.BCELoss(weight=verb_weight)

        if self.config.VOTE_COMB == "att":
            self.mh_att = MHAtt(dim=self.config.EMBEDDING_DIM)
            self.vote_final = three_layer_mlp(
                self.config.NUM_RULE * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
            )

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
        if lens is None:
            if self.config.get("OR_SHUFFLE", False) and self.is_training:
                rand_idx = torch.randperm(input.shape[-2])
                input = input[..., rand_idx, :]
            output = input[..., 0, :]
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))
        else:
            if self.config.get("OR_SHUFFLE", False):
                raise ValueError("Cannot shuffle when rule varies!")

            # Save every output to hidden
            hidden = torch.empty_like(input)
            output = input[..., 0, :]
            hidden[..., 0, :] = output
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))

                hidden[..., i, :] = output

            # Index hidden with lens
            hidden = hidden.view(-1, *input.shape[-2:])  # (bz*18*10,k,embed_dim)
            output = hidden[range(hidden.shape[0]), lens.view(-1) - 1, :]
            output = output.view(*input.shape[:-2], input.shape[-1])

        return output

    def logic_loss_x_or_T(self, x):
        # x or T == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand, self.or_layers(torch.cat((x, T_expand), dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_F(self, x):
        # x or F == x
        F_expand = self.not_module(self.T.unsqueeze(0)).expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, F_expand], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_x(self, x):
        # x or x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, x], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_not_x(self, x):
        # x or (not x) == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand,
                self.or_layers(torch.cat([x, self.not_module(x)], dim=-1)),
                dim=-1,
            )
        )
        return reg

    def logic_loss_not_x(self, x):
        # logical_regularizer, not
        # not+T == F
        reg = 1 + torch.mean(torch.cosine_similarity(x, self.not_module(x), dim=-1))
        return reg

    def logic_loss_not_not_x(self, x):
        # logical_regularizer, not
        # not+not+x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(x, self.not_module(self.not_module(x)), dim=-1)
        )
        return reg

    def logic_loss_new(self):
        reg_list = {
            "pvp_L": self.pvp_L,
            "pos_input": self.pos_input,
            "not_output": self.not_output,
            "or_76_output": self.or_76_output,
            "or_77_output": self.or_77_output,
            "hoi_L": self.hoi_L,
            "or_vote_output": self.or_vote_output,
        }
        reg_targets = [reg_list.get(x) for x in self.config.REG_LIST]

        reg_1 = sum(self.logic_loss_not_x(reg_target) for reg_target in reg_targets)
        reg_2 = sum(self.logic_loss_not_not_x(reg_target) for reg_target in reg_targets)
        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def logic_loss(self, batch):
        # logical_regularizer, not
        # not+T == F
        # (...,embed_dim)->(-1,embed_dim)
        if not self.config.get("ADD_NON", False):
            reg_1 = (1 + torch.cosine_similarity(self.pos_input, self.not_output, dim=-1).mean())
            # not+not+x == x
            not_not_output = self.not_module(self.not_output)
            reg_2 = (1 - torch.cosine_similarity(self.pos_input, not_not_output, dim=-1).mean())
        else:
            pvp_L_neg = self.not_module(self.pvp_L_pos)
            reg_1 = (1 + torch.cosine_similarity(self.pvp_L_pos, pvp_L_neg, dim=-1).mean())
            reg_2 = (1 - torch.cosine_similarity(self.pvp_L_pos, self.not_module(pvp_L_neg), dim=-1).mean())

        # logical_regularizer, or
        # x or T == T
        # not_output(bz,18,10,k,embed_dim), or_77_output(bz,18,10,embed_dim), hoi_L(bz,18,10,embed_dim)
        or_reg_targets = [self.not_output, self.or_77_output, self.hoi_L]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in or_reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in or_reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in or_reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in or_reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def forward(self, batch):
        # batch['rule'] : (bz,18,10,k)
        # 18 rule selected for each pair(according to gt_obj); k/76 pos where rule=1, index, currently k=22
        # batch['gt_label']: (bz,18)
        # batch['gt_range']: (bz,18)
        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.or_layers = self.or_module
        batch_size = batch["rule"].shape[0]

        if hasattr(self, 'pvp_calib'):
            gt_pvp = self.pvp_calib(batch['gt_pvp'])
        else:
            gt_pvp = batch['gt_pvp']

        if not self.config.get("ADD_NON", False):
            # (bz,76),(76,1576)->(bz,76,1576)
            self.pvp_L = torch.einsum("ij,jk->ijk", gt_pvp, self.pvp_matrix)
            # if self.config.get('ADD_NON', False):
            #     self.pvp_L += torch.einsum('ij,jk->ijk', 1 - batch['gt_pvp'], self.non_matrix)
            # (bz,80)*(80,768)=(bz,768)
            if self.config.NUM_CLASS == 600:
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
        else:
            self.pvp_L = self.pvp_matrix.expand(batch_size, -1, -1)
            # (bz,80)*(80,768)=(bz,768)
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
            self.pvp_L_pos = self.pvp_L.clone()
            self.pvp_L = gt_pvp[..., None] * self.pvp_L + (
                    1 - gt_pvp[..., None]
            ) * self.not_module(self.pvp_L)

        if batch["rule"].shape[-1] < self.config.NUM_PVP:
            # input rule is index where label=1
            # batch_idx: (bz,)->(bz,1,1,1)->(bz,18,10,k)
            batch_idx = (
                torch.arange(batch_size).view(-1, 1, 1, 1).expand(batch["rule"].shape)
            )
            # batch_idx: (bz,18,10,k), batch['rule']: (bz,18,10,k), pvp_L: (bz,76,embed_dim)
            # pos_input: (bz,18,10,k,embed_dim)
            if not self.config.get("DYNAMIC_RULE", False):
                self.pos_input = self.pvp_L[batch_idx, batch["rule"], :]
                self.not_output = self.not_module(self.pos_input)
                # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
                self.or_76_output = self.or_module(self.not_output)
            else:
                # # This part find unique value in rule and padded back to original shape with 0
                self.pos_input = self.pvp_L[batch_idx, batch["var_rule"], :]
                self.not_output = self.not_module(self.pos_input)
                # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
                self.or_76_output = self.or_module(self.not_output, batch["rule_lens"])

        # hoi_matrix: (600,1536)->(bz,18,1536)
        self.hoi_L = self.hoi_matrix[batch["gt_range"], :]
        if self.config.NUM_CLASS == 600:
            self.hoi_L = torch.cat(
                [
                    self.hoi_L,
                    self.obj_L.unsqueeze(1).expand(
                        -1, self.config.NUM_CLASS_SELECT, -1
                    ),
                ],
                dim=2,
            )  # (bz,18,2304)
        # (bz,76,2304)->(bz,76,embed_dim)
        self.hoi_L = self.hoi_embedding(self.hoi_L)
        self.hoi_L = self.hoi_L.unsqueeze(2).expand(
            -1, -1, self.config.NUM_RULE, -1
        )  # (bz,18,10,embed_dim)

        # (bz,18,10,2*embed_dim)->(or layer)->(bz,18,10,embed_dim)
        self.or_77_output = self.or_layers(
            torch.cat([self.hoi_L, self.or_76_output], dim=-1)
        )

        if self.config.VOTE_COMB == "or":
            # (bz,18,10,embed_dim)->(or module)->(bz,18,embed_dim)
            self.or_vote_output = self.or_module(self.or_77_output)
            # (bz,18,embed_dim)->(bz,18), s: [-1,1]
        elif self.config.VOTE_COMB == "att":
            # or_77_output: (bz,18,10,embed_dim)
            mh_att = self.mh_att(self.or_77_output)
            self.or_vote_output = self.vote_final(
                mh_att.reshape(*self.or_77_output.shape[:2], -1)
            )
        else:
            raise NotImplementedError

        self.s = torch.cosine_similarity(
            self.or_vote_output,
            self.T[None, None, :],
            dim=-1,
            eps=self.config.get("COS_EPS", 1e-8),
        )

        self.p = self.probability(self.s)
        # self.p = (self.s + 1) / 2

        output = {}
        output["s"] = self.s
        output["p"] = self.p

        if "gt_label" in batch.keys():
            if self.config.LOSS.MODE == "minus":
                # self.s, (bz,18)->(bz)->(bz,18)
                x = torch.sum(self.s * batch["gt_label"], dim=-1)[:, None] - self.s
                output["L_cls"] = torch.mean(
                    -torch.log(torch.sigmoid(torch.sum(x, dim=-1)))
                )
            elif self.config.LOSS.MODE == "BCE":
                assert not torch.isnan((self.s + 1) / 2).any()
                s_ = torch.max((self.s + 1) / 2, torch.zeros(self.s.shape).cuda())
                s_ = torch.min(s_, torch.ones(self.s.shape).cuda())
                output["L_cls"] = self.classification_loss(s_, batch["gt_label"])
            else:
                output["L_cls"] = self.classification_loss(
                    (self.s + 1) / 2, batch["gt_label"].long()
                )

            if self.config.get("REG_LIST"):
                output["L_logic"] = self.logic_loss_new()
            else:
                output["L_logic"] = self.logic_loss(batch)

            output["loss"] = (
                    self.config.LOSS.CLS_FAC * output["L_cls"]
                    + self.config.LOSS.LOGIC_FAC * output["L_logic"]
            )

        return output


class NLR_simplified(nn.Module):
    def __init__(self, config, hoi_weight):
        super(NLR_simplified, self).__init__()

        # simplified model based on the setting in GT-53 (Wed_Jul_15_09:35:46_2020,mAP=38.92), delete some conditional judgement
        self.config = config.MODEL
        self.is_training = True

        self.T = torch.randn(self.config.EMBEDDING_DIM).cuda()

        self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        self.obj_matrix = np.array(obj_matrix, dtype="float32")
        self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)
        # self.hoi_matrix = np.array(hoi_matrix, dtype="float32")
        self.hoi_matrix = np.array(hoi_matrix_2304, dtype="float32")
        self.hoi_matrix = torch.from_numpy(self.hoi_matrix).cuda()  # (600,1536)

        self.pvp_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.hoi_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.not_layers = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
        )

        self.OR_LAYER_TYPE = "rnn"
        if self.config.get("OR_LAYER_TYPE", "rnn") == "rnn":
            self.or_layers = three_layer_mlp(
                2 * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                act=self.config.get("OR_ACT", "relu"),
                dropout=self.config.get("DROPOUT", 0.0),
            )
        elif self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.OR_LAYER_TYPE = "att"
            self.or_attn = MHAtt(dim=self.config.EMBEDDING_DIM)
        else:
            raise NotImplementedError

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCELoss()

        if self.config.VOTE_COMB == "att":
            self.mh_att = MHAtt(dim=self.config.EMBEDDING_DIM)
            self.vote_final = three_layer_mlp(
                self.config.NUM_RULE * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
            )

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
        if lens is None:
            if self.config.get("OR_SHUFFLE", False) and self.is_training:
                rand_idx = torch.randperm(input.shape[-2])
                input = input[..., rand_idx, :]
            output = input[..., 0, :]
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))
        else:
            if self.config.get("OR_SHUFFLE", False):
                raise ValueError("Cannot shuffle when rule varies!")

            # Save every output to hidden
            hidden = torch.empty_like(input)
            output = input[..., 0, :]
            hidden[..., 0, :] = output
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))

                hidden[..., i, :] = output

            # Index hidden with lens
            hidden = hidden.view(-1, *input.shape[-2:])  # (bz*18*10,k,embed_dim)
            output = hidden[range(hidden.shape[0]), lens.view(-1) - 1, :]
            output = output.view(*input.shape[:-2], input.shape[-1])

        return output

    def logic_loss_x_or_T(self, x):
        # x or T == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand, self.or_layers(torch.cat((x, T_expand), dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_F(self, x):
        # x or F == x
        F_expand = self.not_module(self.T.unsqueeze(0)).expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, F_expand], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_x(self, x):
        # x or x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, x], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_not_x(self, x):
        # x or (not x) == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand,
                self.or_layers(torch.cat([x, self.not_module(x)], dim=-1)),
                dim=-1,
            )
        )
        return reg

    def logic_loss_not_x(self, x):
        # logical_regularizer, not
        # not+T == F
        reg = 1 + torch.mean(torch.cosine_similarity(x, self.not_module(x), dim=-1))
        return reg

    def logic_loss_not_not_x(self, x):
        # logical_regularizer, not
        # not+not+x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(x, self.not_module(self.not_module(x)), dim=-1)
        )
        return reg

    def logic_loss_new(self):
        reg_list = {
            "pvp_L": self.pvp_L,
            "pos_input": self.pos_input,
            "not_output": self.not_output,
            "or_76_output": self.or_76_output,
            "or_77_output": self.or_77_output,
            "hoi_L": self.hoi_L,
            "or_vote_output": self.or_vote_output,
        }
        reg_targets = [reg_list.get(x) for x in self.config.REG_LIST]

        reg_1 = sum(self.logic_loss_not_x(reg_target) for reg_target in reg_targets)
        reg_2 = sum(self.logic_loss_not_not_x(reg_target) for reg_target in reg_targets)
        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def logic_loss(self, batch):
        # logical_regularizer, not
        # not+T == F
        # (...,embed_dim)->(-1,embed_dim)
        reg_1 = (1 + torch.cosine_similarity(self.pos_input, self.not_output, dim=-1).mean())
        # not+not+x == x
        not_not_output = self.not_module(self.not_output)
        reg_2 = (1 - torch.cosine_similarity(self.pos_input, not_not_output, dim=-1).mean())

        # logical_regularizer, or
        # x or T == T
        # not_output(bz,18,10,k,embed_dim), or_77_output(bz,18,10,embed_dim), hoi_L(bz,18,10,embed_dim)
        or_reg_targets = [self.not_output, self.or_77_output, self.hoi_L]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in or_reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in or_reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in or_reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in or_reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def forward(self, batch):
        # with open('exp/T.pkl', 'wb') as f:
        #     pickle.dump(self.T.detach().cpu().numpy(), f)

        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.or_layers = self.or_module
        batch_size = batch["rule"].shape[0]

        # (bz,76),(76,1576)->(bz,76,1576)
        self.pvp_L = torch.einsum("ij,jk->ijk", batch["gt_pvp"], self.pvp_matrix)
        # if self.config.get('ADD_NON', False):
        #     self.pvp_L += torch.einsum('ij,jk->ijk', 1 - batch['gt_pvp'], self.non_matrix)
        # (bz,80)*(80,768)=(bz,768)
        if self.config.NUM_CLASS == 600:
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

        if not self.config.get("DYNAMIC_RULE", False):
            self.pos_input = self.pvp_L[batch_idx, batch["rule"], :]
            self.not_output = self.not_module(self.pos_input)
            # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
            self.or_76_output = self.or_module(self.not_output)
        else:
            # # This part find unique value in rule and padded back to original shape with 0
            self.pos_input = self.pvp_L[batch_idx, batch["var_rule"], :]
            self.not_output = self.not_module(self.pos_input)
            # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
            self.or_76_output = self.or_module(self.not_output, batch["rule_lens"])

        # hoi_matrix: (600,1536)->(bz,18,1536)
        self.hoi_L = self.hoi_matrix[batch["gt_range"], :]
        # if self.config.NUM_CLASS == 600:
        #     self.hoi_L = torch.cat(
        #         [
        #             self.hoi_L,
        #             self.obj_L.unsqueeze(1).expand(
        #                 -1, self.config.NUM_CLASS_SELECT, -1
        #             ),
        #         ],
        #         dim=2,
        #     )  # (bz,18,2304)
        # (bz,76,2304)->(bz,76,embed_dim)
        self.hoi_L = self.hoi_embedding(self.hoi_L)
        self.hoi_L = self.hoi_L.unsqueeze(2).expand(
            -1, -1, self.config.NUM_RULE, -1
        )  # (bz,18,10,embed_dim)

        # (bz,18,10,2*embed_dim)->(or layer)->(bz,18,10,embed_dim)
        self.or_77_output = self.or_layers(
            torch.cat([self.hoi_L, self.or_76_output], dim=-1)
        )

        if self.config.VOTE_COMB == "or":
            # (bz,18,10,embed_dim)->(or module)->(bz,18,embed_dim)
            self.or_vote_output = self.or_module(self.or_77_output)
            # (bz,18,embed_dim)->(bz,18), s: [-1,1]
        elif self.config.VOTE_COMB == "att":
            # or_77_output: (bz,18,10,embed_dim)
            mh_att = self.mh_att(self.or_77_output)
            self.or_vote_output = self.vote_final(
                mh_att.reshape(*self.or_77_output.shape[:2], -1)
            )
        else:
            raise NotImplementedError

        self.s = torch.cosine_similarity(
            self.or_vote_output,
            self.T[None, None, :],
            dim=-1,
            eps=self.config.get("COS_EPS", 1e-8),
        )

        self.p = self.probability(self.s)

        output = {}
        output["s"] = self.s
        output["p"] = self.p

        if "gt_label" in batch.keys():
            assert not torch.isnan((self.s + 1) / 2).any()
            s_ = torch.max((self.s + 1) / 2, torch.zeros(self.s.shape).cuda())
            s_ = torch.min(s_, torch.ones(self.s.shape).cuda())
            output["L_cls"] = self.classification_loss(s_, batch["gt_label"])

            if self.config.get("REG_LIST"):
                output["L_logic"] = self.logic_loss_new()
            else:
                output["L_logic"] = self.logic_loss(batch)

            output["loss"] = (
                    self.config.LOSS.CLS_FAC * output["L_cls"]
                    + self.config.LOSS.LOGIC_FAC * output["L_logic"]
            )

        return output


class VL_align(nn.Module):
    def __init__(self, config):
        super(VL_align, self).__init__()

        self.config = config

        self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        self.obj_matrix = np.array(obj_matrix, dtype="float32")
        self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)

        self.part_embedding = three_layer_mlp(
            self.config.VISUAL_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
        )

        self.pvp_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
        )

        self.LogSigmoid = torch.nn.LogSigmoid()
        self.P_num = [12, 10, 5, 31, 5, 13]
        self.P_index = np.concatenate([[0], np.cumsum(self.P_num)])  # [0 12 22 27 58 63 76]
        self.P_mapping_6v_to_10v = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
        self.P_keys = ['P0l', 'P0r', 'P1l', 'P1r', 'P2', 'P3l', 'P3r', 'P4l', 'P4r', 'P5']
        # [0,12), [12,22), [22,27), [27,58), [58,63), [63,76)

        self.load_pretrain(self.config.ALIGN.CHECKPOINT)

    def load_pretrain(self, pretrain_path):
        pretrained_dict = torch.load(pretrain_path)['state'].state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        for k, v in self.named_parameters():
            if k in pretrained_dict.keys():
                v.requires_grad = False
        print('Align: Parameters frozen.')

        for k, v in self.named_parameters():
            print(k, v.requires_grad)

        print('Align: Pretrained loaded.')

    def forward(self, batch):

        self.part_V = torch.cat(
            [batch[key].unsqueeze(1)
             for key in
             ['FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5']],
            dim=1)  # (bz,10,1024)

        self.part_V = batch['gt_part'].unsqueeze(-1).expand(
            self.part_V.shape) * self.part_V  # (bz,10)*(bz,10,1024)=(bz,10,1024)

        for F in ['FO', 'FS', 'FH']:
            if F in self.config.F_keys:
                self.part_V = torch.cat(
                    [self.part_V, batch[F].unsqueeze(1).expand(self.part_V.shape)],
                    dim=-1)  # (bz,10,VISUAL_DIM)

        self.part_V = self.part_embedding(self.part_V)  # (bz,10,VISUAL_DIM)->(bz,10,EMBEDDING_DIM)

        gt_pvp = torch.cat([batch[key] for key in self.config.L_keys[:-1]], dim=-1).cuda()
        # print(gt_pvp.shape)

        self.pvp_L = torch.einsum("ij,jk->ijk", gt_pvp, self.pvp_matrix)

        self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
        self.pvp_L = torch.cat(
            [
                self.pvp_L,
                self.obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1),
            ],
            dim=2,
        )  # (bz,76,2304)

        self.pvp_L = self.pvp_embedding(self.pvp_L)  # (bz,76,2304)->(bz,76,EMBEDDING_DIM)

        # (bz,EMBEDDING_DIM), (bz,Pasta_nums,EMBEDDING_DIM) , batch["gt_pvp"] (bz,76)
        # self.part_V (bz,EMBEDDING_DIM) * self.pvp_L (bz,Pasta_nums,EMBEDDING_DIM) -> (bz,Pasta_nums,EMBEDDING_DIM)
        # (bz,Pasta_nums,EMBEDDING_DIM) * 1 - batch["gt_pvp"] (bz,Pasta_nums) -> (bz,Pasta_nums,EMBEDDING_DIM) -> (-) -> log sigmoid -> sum
        # A and B and C = not-(not-A or not-B or not-C)
        # (bz,Pasta_nums,EMBEDDING_DIM) * batch["gt_pvp"] ->(and)->(bz,EMBEDDING_DIM) -> log sigmoid -> sum
        output = {}

        for i in range(self.part_V.shape[-2]):
            # self.cur_part_V (bz,EMBEDDING_DIM); self.cur_pvp_L (bz,Pasta_nums,EMBEDDING_DIM)
            self.cur_part_V = self.part_V[..., i, :]
            self.cur_pvp_L = self.pvp_L[...,
                             self.P_index[self.P_mapping_6v_to_10v[i]]:self.P_index[self.P_mapping_6v_to_10v[i] + 1], :]
            cur_gt_pvp = gt_pvp[...,
                         self.P_index[self.P_mapping_6v_to_10v[i]]:self.P_index[self.P_mapping_6v_to_10v[i] + 1]]
            if not self.config.ALIGN.POS_AND:
                dis = torch.einsum('ij,ikj->ik', self.cur_part_V, self.cur_pvp_L)  # (bz,Pasta_nums)
                neg_dis = dis * (1 - cur_gt_pvp)
                pos_dis = dis * cur_gt_pvp

                output[self.P_keys[i]] = -torch.where(cur_gt_pvp > 0,
                                                      self.LogSigmoid(pos_dis), self.LogSigmoid(-neg_dis)).mean()
            else:
                print('POS_AND: True.')

        output["loss"] = sum([output[k] for k in self.P_keys])
        output["visual_embed"] = self.part_V

        return output


class Linear_10v(nn.Module):
    def __init__(self, config, hoi_weight):
        super(Linear_10v, self).__init__()
        self.config = config.MODEL

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=hoi_weight)

        self.linear = nn.Linear(
            10 * self.config.VISUAL_DIM, self.config.NUM_CLASS
        )

    def forward(self, batch):
        self.part_V = torch.cat(
            [batch[key].unsqueeze(1)
             for key in
             ['FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5']],
            dim=1)  # (bz,10,P_dim)

        self.part_V = batch['gt_part'].unsqueeze(-1).expand(
            self.part_V.shape) * self.part_V  # (bz,10)*(bz,10,P_dim)=(bz,10,P_dim)

        for F in ['FO', 'FH']:
            if F in self.config.F_keys:
                self.part_V = torch.cat(
                    [self.part_V, batch[F].unsqueeze(1).expand(self.part_V.shape)],
                    dim=-1)  # (bz,10,2048)

        s = self.linear(self.part_V.flatten(1, 2))

        p = self.probability(s)

        output = {}
        output["s"] = s
        output["p"] = p

        if "labels_r" in batch.keys():
            output["L_cls"] = self.classification_loss(s, batch["labels_r"])
            output["loss"] = output["L_cls"]

        return output



class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


acti = {
    'ReLU': nn.ReLU,
    'Swish': Swish,
}


class pvp_classifier(nn.Module):  # load from 1500: pasta_torch
    def __init__(self, config, pvp_weight, frozen):

        super(pvp_classifier, self).__init__()
        self.config = config
        self.encoder = OrderedDict()

        for i in range(1, len(config.LAYER_SIZE)):
            self.encoder['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.encoder['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.encoder['relu%d' % i] = acti[config.ACT]()
        self.encoder = nn.Sequential(self.encoder)

        self.decoder = OrderedDict()
        for i in reversed(range(1, len(config.LAYER_SIZE))):
            self.decoder['fc%d' % (len(config.LAYER_SIZE) - i)] = nn.Linear(config.LAYER_SIZE[i],
                                                                            config.LAYER_SIZE[i - 1])
            if i > 1:
                if config.BN:
                    self.decoder['bn%d' % (len(config.LAYER_SIZE) - i)] = nn.BatchNorm1d(config.LAYER_SIZE[i - 1])
                self.decoder['relu%d' % (len(config.LAYER_SIZE) - i)] = acti[config.ACT]()
        self.decoder = nn.Sequential(self.decoder)

        self.key = config.KEY
        self.classifier = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pvp_weight[self.key[1]])
        self.rec_loss = nn.MSELoss()
        self.cls_fac = config.CLS_FAC
        self.rec_fac = config.REC_FAC
        if config.CHECKPOINT:
            self.load_state_dict(torch.load(config.CHECKPOINT)['state'])

        if frozen:
            for k, v in self.named_parameters():
                v.requires_grad = False

    def forward(self, batch):
        if batch['FR'].shape[-1] == 2048:
            x = torch.cat([batch[self.key[0]], batch['FH'], batch['FR'], batch['FS']], dim=1)
        elif batch['FR'].shape[-1] == 1024:
            x = torch.cat([batch[self.key[0]], batch['FH'], batch['FR'], batch['FS'], batch['FS']], dim=1)
        else:
            raise NotImplementedError

        z = self.encoder(x)  # fc7_P
        x_ = self.decoder(z)
        s = self.classifier(z)
        L_cls = self.cls_fac * self.cls_loss(s, batch[self.key[1]])
        L_rec = self.rec_fac * self.rec_loss(x, x_)
        loss = L_cls + L_rec
        p = torch.sigmoid(s)

        output = {
            'z': z,
            's': s,
            'p': p,
            'L_cls': L_cls,
            'L_rec': L_rec,
            'loss': loss
        }

        return output


class part_classifier(nn.Module):  # load from 1500: pasta_torch
    def __init__(self, config, pvp_weight):
        super(part_classifier, self).__init__()
        self.config = config
        self.P0_classifier = pvp_classifier(config.P0_CLS, pvp_weight, config.FROZEN)
        self.P1_classifier = pvp_classifier(config.P1_CLS, pvp_weight, config.FROZEN)
        self.P2_classifier = pvp_classifier(config.P2_CLS, pvp_weight, config.FROZEN)
        self.P3_classifier = pvp_classifier(config.P3_CLS, pvp_weight, config.FROZEN)
        self.P4_classifier = pvp_classifier(config.P4_CLS, pvp_weight, config.FROZEN)
        self.P5_classifier = pvp_classifier(config.P5_CLS, pvp_weight, config.FROZEN)

        self.use_att = config.get('USE_ATT', False)

    def forward(self, batch):
        out_P0 = self.P0_classifier(batch)
        out_P1 = self.P1_classifier(batch)
        out_P2 = self.P2_classifier(batch)
        out_P3 = self.P3_classifier(batch)
        out_P4 = self.P4_classifier(batch)
        out_P5 = self.P5_classifier(batch)

        if self.use_att:
            z = torch.cat([
                out_P0['z'] * torch.max(out_P0['p'][:, :-1], dim=-1, keepdim=True).values,
                out_P1['z'] * torch.max(out_P1['p'][:, :-1], dim=-1, keepdim=True).values,
                out_P2['z'] * torch.max(out_P2['p'][:, :-1], dim=-1, keepdim=True).values,
                out_P3['z'] * torch.max(out_P3['p'][:, :-1], dim=-1, keepdim=True).values,
                out_P4['z'] * torch.max(out_P4['p'][:, :-1], dim=-1, keepdim=True).values,
                out_P5['z'] * torch.max(out_P5['p'][:, :-1], dim=-1, keepdim=True).values], dim=1)
        else:
            z = torch.cat([out_P0['z'], out_P1['z'], out_P2['z'], out_P3['z'], out_P4['z'], out_P5['z']], dim=1)

        # if self.config.SCORE_MODE == 0:
        if not self.config.get('MAX_PASTA') or not self.config.MAX_PASTA:
            s_10v = torch.cat(
                [1 - out_P0['s'][..., -1].unsqueeze(-1), 1 - out_P0['s'][..., -1].unsqueeze(-1),
                 1 - out_P1['s'][..., -1].unsqueeze(-1), 1 - out_P1['s'][..., -1].unsqueeze(-1),
                 1 - out_P2['s'][..., -1].unsqueeze(-1),
                 1 - out_P3['s'][..., -1].unsqueeze(-1), 1 - out_P3['s'][..., -1].unsqueeze(-1),
                 1 - out_P4['s'][..., -1].unsqueeze(-1), 1 - out_P4['s'][..., -1].unsqueeze(-1),
                 1 - out_P5['s'][..., -1].unsqueeze(-1)], dim=-1)
        else:
            s_10v = torch.cat([torch.max(out_P0['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P0['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P1['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P1['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P2['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P3['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P3['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P4['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P4['s'][..., :-1], dim=-1)[0].unsqueeze(-1),
                               torch.max(out_P5['s'][..., :-1], dim=-1)[0].unsqueeze(-1)], dim=-1)

        output = {
            's': torch.cat([out_P0['s'], out_P1['s'], out_P2['s'], out_P3['s'], out_P4['s'], out_P5['s']], dim=1),
            # cls_score_P0 ~ cls_score_P5
            'z': z,  # fc7_P
            's_10v': s_10v,
            'l0': out_P0['loss'],
            'l1': out_P1['loss'],
            'l2': out_P2['loss'],
            'l3': out_P3['loss'],
            'l4': out_P4['loss'],
            'l5': out_P5['loss'],
        }
        if 'labels_r' in batch:
            output['labels_r'] = batch['labels_r']
        if 'labels_v' in batch:
            output['labels_v'] = batch['labels_v']

        return output


class NLR_10v(nn.Module):
    def __init__(self, config, pvp_weight):
        super(NLR_10v, self).__init__()

        # based on NLR_simplified model

        self.config = config.MODEL.NLR
        self.is_training = True

        if not self.config.GT_PART or self.config.FEATURE == 'encode':
            self.part_classifier = part_classifier(config.MODEL.PART_CLS, pvp_weight)

        self.T = torch.randn(self.config.EMBEDDING_DIM).cuda()

        self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        self.obj_matrix = np.array(obj_matrix, dtype="float32")
        self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)

        self.hoi_matrix = np.array(hoi_matrix_2304, dtype="float32")
        # self.hoi_matrix = np.array(hoi_matrix, dtype="float32")
        self.hoi_matrix = torch.from_numpy(self.hoi_matrix).cuda()  # (600,1536)

        self.partV_embedding = three_layer_mlp(
            self.config.VISUAL_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        if self.config.ALIGN:
            self.pvp_embedding = three_layer_mlp(
                self.config.BERT_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
            )

        if self.config.NUM_PART == 6 and self.config.VISUAL_DIM == 2048:
            self.FP0_encoder, self.FP1_encoder, self.FP3_encoder, self.FP4_encoder = (three_layer_mlp(2048, 1024, 1024)
                                                                                      for _ in range(4))

        self.LogSigmoid = torch.nn.LogSigmoid()
        self.P_num = [12, 10, 5, 31, 5, 13]
        self.P_index = np.concatenate([[0], np.cumsum(self.P_num)])  # [0 12 22 27 58 63 76]
        self.P_mapping_6v_to_10v = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
        self.P_keys = ['P0l', 'P0r', 'P1l', 'P1r', 'P2', 'P3l', 'P3r', 'P4l', 'P4r', 'P5']
        # [0,12), [12,22), [22,27), [27,58), [58,63), [63,76)

        self.hoiV_embedding = three_layer_mlp(
            self.config.HOIV_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )
        self.hoi_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.not_layers = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
        )

        self.OR_LAYER_TYPE = "rnn"
        if self.config.get("OR_LAYER_TYPE", "rnn") == "rnn":
            self.or_layers = three_layer_mlp(
                2 * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                act=self.config.get("OR_ACT", "relu"),
                dropout=self.config.get("DROPOUT", 0.0),
            )
        elif self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.OR_LAYER_TYPE = "att"
            self.or_attn = MHAtt(dim=self.config.EMBEDDING_DIM)
        else:
            raise NotImplementedError

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCELoss()

        if self.config.VOTE_COMB == "att":
            self.mh_att = MHAtt(dim=self.config.EMBEDDING_DIM)
            self.vote_final = three_layer_mlp(
                self.config.NUM_RULE * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
            )

        self.load_pretrain()

    def load_pretrain(self):
        if not self.config.CHECKPOINT:
            if self.config.ALIGN:
                return ValueError
            return
        pretrained_dict = torch.load(self.config.CHECKPOINT)['state'].state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        T_path = self.config.CHECKPOINT[:-7] + 'T.pkl'
        print(T_path)
        if os.path.exists(T_path):
            T = pickle.load(open(T_path, 'rb'))
            self.T = torch.from_numpy(T).cuda()
            print(self.T)
        else:
            print('T not exist!')

        if self.config.FROZEN:
            for key in self.config.FROZEN:
                for k, v in self.named_parameters():
                    if key in k:
                        v.requires_grad = False

        for k, v in self.named_parameters():
            print(k, v.requires_grad)

        print('pretrained loaded.')

    def set_status(self, is_training=True):
        self.is_training = is_training

    def not_module(self, input):
        return self.not_layers(input)

    def or_module(self, input, lens=None):
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

        if lens is None:
            if self.config.get("OR_SHUFFLE", False) and self.is_training:
                rand_idx = torch.randperm(input.shape[-2])
                input = input[..., rand_idx, :]
            output = input[..., 0, :]
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))
        else:
            if self.config.get("OR_SHUFFLE", False):
                raise ValueError("Cannot shuffle when rule varies!")

            # Save every output to hidden
            # a little different from DYNAMIC_RULE in NLR

            hidden = torch.zeros((*input.shape[:-2], input.shape[-2] + 1, input.shape[-1])).cuda()
            output = input[..., 0, :]
            hidden[..., 1, :] = output

            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))

                hidden[..., i + 1, :] = output

            # Index hidden with lens

            hidden = hidden.view(-1, *hidden.shape[-2:])  # (bz*18*10,k,embed_dim)
            output = hidden[range(hidden.shape[0]), lens.view(-1), :]
            output = output.view(*input.shape[:-2], input.shape[-1])

        return output

    def logic_loss_x_or_T(self, x):
        # x or T == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand, self.or_layers(torch.cat((x, T_expand), dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_F(self, x):
        # x or F == x
        F_expand = self.not_module(self.T.unsqueeze(0)).expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, F_expand], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_x(self, x):
        # x or x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                x, self.or_layers(torch.cat([x, x], dim=-1)), dim=-1
            )
        )
        return reg

    def logic_loss_x_or_not_x(self, x):
        # x or (not x) == T
        # T(embed_dim) -> T(x.shape)
        T_expand = self.T.expand(x.shape)
        reg = 1 - torch.mean(
            torch.cosine_similarity(
                T_expand,
                self.or_layers(torch.cat([x, self.not_module(x)], dim=-1)),
                dim=-1,
            )
        )
        return reg

    def logic_loss_not_x(self, x):
        # logical_regularizer, not
        # not+T == F
        reg = 1 + torch.mean(torch.cosine_similarity(x, self.not_module(x), dim=-1))
        return reg

    def logic_loss_not_not_x(self, x):
        # logical_regularizer, not
        # not+not+x == x
        reg = 1 - torch.mean(
            torch.cosine_similarity(x, self.not_module(self.not_module(x)), dim=-1)
        )
        return reg

    def logic_loss(self):
        reg_1 = (1 + torch.cosine_similarity(self.pos_input, self.not_output, dim=-1).mean())
        # not+not+x == x
        not_not_output = self.not_module(self.not_output)
        reg_2 = (1 - torch.cosine_similarity(self.pos_input, not_not_output, dim=-1).mean())

        or_reg_targets = [self.not_output, self.or_10_output, self.or_11_output, self.hoi]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in or_reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in or_reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in or_reg_targets)
        reg_6 = sum(
            self.logic_loss_x_or_not_x(reg_target) for reg_target in or_reg_targets
        )

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def logic_loss_new(self):

        reg_targets = [self.part_V, self.pos_input, self.not_output, self.or_10_output, self.or_11_output, self.hoi]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

        reg_1 = sum(self.logic_loss_not_x(reg_target) for reg_target in reg_targets)
        reg_2 = sum(self.logic_loss_not_not_x(reg_target) for reg_target in reg_targets)
        reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in reg_targets)
        reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in reg_targets)
        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in reg_targets)
        reg_6 = sum(self.logic_loss_x_or_not_x(reg_target) for reg_target in reg_targets)

        dic = {1: reg_1, 2: reg_2, 3: reg_3, 4: reg_4, 5: reg_5, 6: reg_6}
        reg = sum(dic[k] for k in self.config.LOGIC_REG)

        return reg

    def forward(self, batch):
        # with open('exp/T.pkl', 'wb') as f:
        #     pickle.dump(self.T.detach().cpu().numpy(), f)

        # batch['rule'] : (bz,18,10,k)
        # batch['gt_label']: (bz,18)
        # batch['gt_range']: (bz,18)
        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.or_layers = self.or_module

        if self.config.ALIGN:
            gt_pvp = torch.cat([batch[key] for key in self.config.L_keys[:-1]], dim=-1).cuda()

            self.pvp_L = torch.einsum("ij,jk->ijk", gt_pvp, self.pvp_matrix)

            self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
            self.pvp_L = torch.cat(
                [
                    self.pvp_L,
                    self.obj_L.unsqueeze(1).expand(-1, self.config.NUM_PVP, -1),
                ],
                dim=2,
            )  # (bz,76,2304)

            self.pvp_L = self.pvp_embedding(self.pvp_L)  # (bz,76,2304)->(bz,76,EMBEDDING_DIM)

        if self.config.FEATURE == 'pooling':
            if self.config.NUM_PART == 6:
                if self.config.VISUAL_DIM == 2048:
                    self.part_V = torch.cat(
                        [self.FP0_encoder(batch['FP0']).unsqueeze(1), self.FP1_encoder(batch['FP1']).unsqueeze(1),
                         batch['FP2'].unsqueeze(1),
                         self.FP3_encoder(batch['FP3']).unsqueeze(1), self.FP4_encoder(batch['FP4']).unsqueeze(1),
                         batch['FP5'].unsqueeze(1)],
                        dim=1)
                else:
                    self.part_V = torch.cat(
                        [batch['FP0'].unsqueeze(1), batch['FP1'].unsqueeze(1),
                         torch.cat([batch['FP2'], batch['FP2']], dim=-1).unsqueeze(1),
                         batch['FP3'].unsqueeze(1), batch['FP4'].unsqueeze(1),
                         torch.cat([batch['FP5'], batch['FP5']], dim=-1).unsqueeze(1)],
                        dim=1)
            else:
                self.part_V = torch.cat(
                    [batch[key].unsqueeze(1)
                     for key in
                     ['FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5']],
                    dim=1)  # (bz,10,P_dim)

            if self.config.GT_PART:
                self.part_V = batch['gt_part'].unsqueeze(-1).expand(
                    self.part_V.shape) * self.part_V  # (bz,10)*(bz,10,P_dim)=(bz,10,P_dim)

            else:
                output_cls = self.part_classifier(batch)
                self.part_V = output_cls['s_10v'].unsqueeze(-1).expand(
                    self.part_V.shape) * self.part_V

            if 'FO' in self.config.F_keys:
                self.part_V = torch.cat(
                    [self.part_V, batch['FO'].unsqueeze(1).expand(*self.part_V.shape[:-2], self.config.NUM_PART, -1)],
                    dim=-1)  # (bz,10,2048)

            self.part_V = self.partV_embedding(self.part_V)  # (bz,10,2048)->(bz,10,embed_dim)

        elif self.config.FEATURE == 'encode':
            output_cls = self.part_classifier(batch)
            self.part_V = output_cls['z']
            self.part_V = self.partV_embedding(self.part_V)

        if self.config.DYNAMIC_RULE:
            batch_idx = (torch.arange(batch["rule"].shape[0]).view(-1, 1, 1, 1).expand(batch["rule"].shape))
            self.pos_input = self.part_V[batch_idx, batch["rule"].long(), :]
            self.not_output = self.not_module(self.pos_input)
            # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
            self.or_10_output = self.or_module(self.not_output, batch["rule_lens"])
        else:
            # (bz,18,10,10), (bz,10,embed_dim)->(bz,18,10,10,embed_dim)
            self.pos_input = torch.einsum('ijkl,ilr->ijklr', batch['rule'], self.part_V)

            if self.config.CAL_MODE == 'one_hot_1':
                self.not_output = self.not_module(self.pos_input)
            elif self.config.CAL_MODE == 'one_hot_2':
                self.not_output = torch.einsum('ijkl,ijklr->ijklr', batch['rule'],
                                               self.not_module(self.pos_input))

            self.or_10_output = self.or_module(
                self.not_output)  # (bz,18,10,10,embed_dim)->(or module)->(bz,18,10,embed_dim)

        if self.config.ACTION_EMBED in ['visual', 'visual_align']:
            self.hoi = self.hoiV_embedding(batch['FR']).unsqueeze(1).unsqueeze(1).expand(
                self.or_10_output.shape)  # (bz,18,10,embed_dim)

        else:
            self.hoi = self.hoi_matrix[batch["gt_range"], :]
            # self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)  # (bz,768)
            # self.hoi = self.hoi_matrix[batch["gt_range"], :]  # (bz,18,1536)
            # self.hoi = torch.cat(
            #     [
            #         self.hoi,
            #         self.obj_L.unsqueeze(1).expand(
            #             -1, self.config.NUM_CLASS_SELECT, -1
            #         ),
            #     ],
            #     dim=2,
            # )  # (bz,18,2304)
            # (bz,76,2304)->(bz,76,embed_dim)
            self.hoi = self.hoi_embedding(self.hoi)
            self.hoi = self.hoi.unsqueeze(2).expand(
                -1, -1, self.config.NUM_RULE, -1
            )  # (bz,18,10,embed_dim)

        self.or_11_output = self.or_layers(
            torch.cat([self.hoi, self.or_10_output], dim=-1))  # (bz,18,10,2*embed_dim)->(bz,18,10,embed_dim)

        mh_att = self.mh_att(self.or_11_output)
        self.or_vote_output = self.vote_final(
            mh_att.reshape(*self.or_11_output.shape[:2], -1)
        )

        self.s = torch.cosine_similarity(
            self.or_vote_output,
            self.T[None, None, :],
            dim=-1,
            eps=self.config.get("COS_EPS", 1e-8),
        )

        self.p = self.probability(self.s)

        output = {}
        output["s"] = self.s
        output["p"] = self.p

        if "labels_r" in batch.keys():
            s_ = torch.max((self.s + 1) / 2, torch.zeros(self.s.shape).cuda())
            s_ = torch.min(s_, torch.ones(self.s.shape).cuda())
            output["L_cls"] = self.classification_loss(s_, batch["labels_r"])

            output["L_logic"] = self.logic_loss()

            output["loss"] = (
                    self.config.LOSS.CLS_FAC * output["L_cls"]
                    + self.config.LOSS.LOGIC_FAC * output["L_logic"]
            )

            if self.config.ACTION_EMBED == 'visual_align':
                hoi_V = self.hoiV_embedding(batch['FR'])  # (bz,embed_dim)

                obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)  # (bz,768)
                hoi_L = self.hoi_matrix[batch["gt_range"], :]  # (bz,18,1536)
                hoi_L = torch.cat(
                    [
                        hoi_L,
                        obj_L.unsqueeze(1).expand(
                            -1, self.config.NUM_CLASS_SELECT, -1
                        ),
                    ],
                    dim=2,
                )  # (bz,18,2304)
                hoi_L = self.hoi_embedding(hoi_L)  # (bz,18,embed_dim)

                dis = torch.einsum('ij,ikj->ik', hoi_V, hoi_L)  # (bz,18)
                neg_dis = dis * (1 - batch['labels_r'])
                pos_dis = dis * batch['labels_r']

                output["L_action"] = -torch.where(batch['labels_r'] > 0,
                                                  self.LogSigmoid(pos_dis),
                                                  self.LogSigmoid(-neg_dis)).mean()

                output["loss"] += self.config.LOSS.ACTION_FAC * output["L_action"]

            if self.config.ALIGN:
                for i in range(self.part_V.shape[-2]):
                    self.cur_part_V = self.part_V[..., i, :]  # self.cur_part_V (bz,EMBEDDING_DIM);
                    self.cur_pvp_L = self.pvp_L[...,
                                     self.P_index[self.P_mapping_6v_to_10v[i]]:self.P_index[
                                         self.P_mapping_6v_to_10v[i] + 1], :]
                    # self.cur_pvp_L (bz,Pasta_nums,EMBEDDING_DIM)

                    cur_gt_pvp = gt_pvp[...,
                                 self.P_index[self.P_mapping_6v_to_10v[i]]:self.P_index[
                                     self.P_mapping_6v_to_10v[i] + 1]]  # (bz,Pasta_nums)

                    dis = torch.einsum('ij,ikj->ik', self.cur_part_V, self.cur_pvp_L)  # (bz,Pasta_nums)
                    neg_dis = dis * (1 - cur_gt_pvp)
                    pos_dis = dis * cur_gt_pvp
                    output[self.P_keys[i]] = -torch.where(cur_gt_pvp > 0,
                                                          self.LogSigmoid(pos_dis),
                                                          self.LogSigmoid(-neg_dis)).mean()

                    # dis = torch.norm(self.cur_part_V.unsqueeze(1).expand(self.cur_pvp_L.shape) - self.cur_pvp_L,
                    #                  dim=-1)  # (bz,Pasta_nums)
                    # neg_dis = dis * (1 - cur_gt_pvp)
                    # pos_dis = dis * cur_gt_pvp
                    # output[self.P_keys[i]] = -torch.where(cur_gt_pvp > 0,
                    #                                       self.LogSigmoid(-pos_dis),
                    #                                       self.LogSigmoid(neg_dis)).mean()

                output["L_align"] = sum([output[k] for k in self.P_keys])
                output["loss"] += self.config.LOSS.ALIGN_FAC * output["L_align"]

        return output


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

        # if self.config.NUM_CLASS in [80, 60] and self.config.NUM_PVP == 76:
        #     self.pvp_embedding = three_layer_mlp(
        #         int(self.config.BERT_DIM * 2 / 3),
        #         self.config.EMBEDDING_DIM,
        #         self.config.EMBEDDING_DIM,
        #     )
        # else:
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

        # for k, v in self.named_parameters():
        #     print(k, v.requires_grad)

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

        # self.gt_pvp = torch.cat([batch[key] for key in self.config.L_keys[:-1]], dim=-1)
        self.gt_pvp = batch["gt_pvp"]

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

        # print(batch_size, batch["rule"].shape)
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
            print(1)

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

            # print('self.or_77_output.shape: ', self.or_77_output.shape)
            # print('vote.shape: ', vote.shape)
            # print(vote.shape)
            # print(len(vote.shape))

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

        # output["logic_embed"] = [
        # self.pos_input.cpu().detach().numpy(),
        # self.or_77_output.cpu().detach().numpy(),
        # self.hoi_L.cpu().detach().numpy()]

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


class NLR_10v_simplified_no_T(nn.Module):
    def __init__(self, config):
        super(NLR_10v_simplified_no_T, self).__init__()

        # based on NLR_simplified model

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

        # self.pvp_matrix = np.array(sentence_only, dtype="float32")[:, :1536]
        # self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (76,1536)
        # self.obj_matrix = np.array(obj_matrix, dtype="float32")
        # self.obj_matrix = torch.from_numpy(self.obj_matrix).cuda()  # (80,768)

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

        self.partV_embedding = three_layer_mlp(
            self.config.VISUAL_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.hoi_embedding = three_layer_mlp(
            self.config.BERT_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
            bn=self.config.get("BN", False),
        )

        self.not_layers = three_layer_mlp(
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            self.config.EMBEDDING_DIM,
            dropout=self.config.get("DROPOUT", 0.0),
        )

        if self.config.get('ADJUST') and self.config.ADJUST:
            self.judger = nn.Linear(
                self.config.EMBEDDING_DIM,
                1,
            )
        else:
            self.judger = three_layer_mlp(
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                1,
            )
        self.judger_loss = nn.BCEWithLogitsLoss()

        self.OR_LAYER_TYPE = "rnn"
        if self.config.get("OR_LAYER_TYPE", "rnn") == "rnn":
            self.or_layers = three_layer_mlp(
                2 * self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                self.config.EMBEDDING_DIM,
                act=self.config.get("OR_ACT", "relu"),
                dropout=self.config.get("DROPOUT", 0.0),
            )
        elif self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.OR_LAYER_TYPE = "att"
            self.or_attn = MHAtt(dim=self.config.EMBEDDING_DIM)
        else:
            raise NotImplementedError

        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCELoss()

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

        self.load_pretrain()

    def load_pretrain(self):
        if not self.config.CHECKPOINT:
            return
        pretrained_dict = torch.load(self.config.CHECKPOINT)['state'].state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        if self.config.get('ADJUST') and self.config.ADJUST:
            # T_path = self.config.CHECKPOINT[:-7] + 'T.pkl'
            # T = pickle.load(open(T_path, 'rb'))
            pretrained_dict['judger.weight'] = torch.from_numpy(
                np.array([[0.2142, -0.1434, -0.2636, 1.2033, 1.0941, 0.7960, 0.0044, -1.5482,
                           -1.1537, -0.1612, 0.6676, -0.4936, 1.6012, 1.3674, -1.9543, 0.0426,
                           -1.6858, -0.3373, 1.6615, 0.4142, -0.7914, 2.2371, -0.2026, 1.0437,
                           0.5246, -2.2965, 0.2583, -2.3315, -0.3692, -1.4560, 0.7281, 1.3429]])).cuda()
            # self.judger.weight[0] = torch.Tensor(T)
            # nn.init.constant_(self.judger.weight, torch.from_numpy(T).cuda())
            nn.init.constant_(self.judger.bias, 0)

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        print('pretrained loaded.')

    def set_status(self, is_training=True):
        self.is_training = is_training

    def not_module(self, input):
        return self.not_layers(input)

    def or_module(self, input, lens=None):
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

        if lens is None:

            if self.config.get("OR_SHUFFLE", False) and self.is_training:
                rand_idx = torch.randperm(input.shape[-2])
                input = input[..., rand_idx, :]
            output = input[..., 0, :]
            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))
        else:
            if self.config.get("OR_SHUFFLE", False):
                raise ValueError("Cannot shuffle when rule varies!")

            # Save every output to hidden
            # a little different from DYNAMIC_RULE in NLR

            hidden = torch.zeros((*input.shape[:-2], input.shape[-2] + 1, input.shape[-1])).cuda()
            output = input[..., 0, :]
            hidden[..., 1, :] = output

            for i in range(1, input.shape[-2]):
                output = self.or_layers(torch.cat((input[..., i, :], output), dim=-1))

                hidden[..., i + 1, :] = output

            # Index hidden with lens
            hidden = hidden.view(-1, *hidden.shape[-2:])  # (bz*18*10,k,embed_dim)
            output = hidden[range(hidden.shape[0]), lens.view(-1), :]
            output = output.view(*input.shape[:-2], input.shape[-1])

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
        or_reg_targets = [self.not_output, self.or_10_output, self.or_11_output, self.hoi_L]

        if self.config.VOTE_COMB == "or":
            or_reg_targets.append(self.or_vote_output)

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
        # batch['gt_label']: (bz,18)
        # batch['gt_range']: (bz,18)
        if self.config.get("OR_LAYER_TYPE", "rnn") == "att":
            self.or_layers = self.or_module

        batch_size = batch["rule"].shape[0]

        self.part_V = torch.cat(
            [batch[key].unsqueeze(1)
             for key in
             ['FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5']],
            dim=1)  # (bz,10,P_dim)

        self.part_V = batch['gt_part'].unsqueeze(-1).expand(
            self.part_V.shape) * self.part_V  # (bz,10)*(bz,10,P_dim)=(bz,10,P_dim)

        if 'FO' in self.config.F_keys:
            self.part_V = torch.cat(
                [self.part_V, batch['FO'].unsqueeze(1).expand(*self.part_V.shape[:-2], self.config.NUM_PART, -1)],
                dim=-1)  # (bz,10,2048)

        self.part_V = self.partV_embedding(self.part_V)  # (bz,10,2048)->(bz,10,embed_dim)

        if self.config.DYNAMIC_RULE:
            batch_idx = (torch.arange(batch["rule"].shape[0]).view(-1, 1, 1, 1).expand(batch["rule"].shape))
            self.pos_input = self.part_V[batch_idx, batch["rule"].long(), :]
            self.not_output = self.not_module(self.pos_input)
            # (bz,18,10,k,embed_dim)->(or module)->(bz,18,10,embed_dim)
            self.or_10_output = self.or_module(self.not_output, batch["rule_lens"])
        else:
            # (bz,18,10,10), (bz,10,embed_dim)->(bz,18,10,10,embed_dim)
            self.pos_input = torch.einsum('ijkl,ilr->ijklr', batch['rule'], self.part_V)
            self.not_output = self.not_module(self.pos_input)
            self.or_10_output = self.or_module(
                self.not_output)  # (bz,18,10,10,embed_dim)->(or module)->(bz,18,10,embed_dim)

        if self.config.NUM_CLASS == 600:
            self.hoi_L = self.hoi_matrix[batch["gt_range"], :]
        elif self.config.NUM_CLASS == 29:
            self.obj_L = torch.matmul(batch["gt_obj"], self.obj_matrix)
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

        self.or_11_output = self.or_layers(
            torch.cat([self.hoi_L, self.or_10_output], dim=-1))  # (bz,18,10,2*embed_dim)->(bz,18,10,embed_dim)

        mh_att = self.mh_att(self.or_11_output)
        self.or_vote_output = self.vote_final(
            mh_att.reshape(*self.or_11_output.shape[:2], -1)
        )

        self.s = self.judger(self.or_vote_output).squeeze()

        self.p = self.probability(self.s)

        output = {}
        output["s"] = self.s
        output["p"] = self.p

        output["logic_embed"] = [
            self.pos_input.cpu().detach().numpy(),
            self.or_11_output.cpu().detach().numpy(),
            self.hoi_L.cpu().detach().numpy()]

        if "gt_label" in batch.keys() or "labels_v" in batch.keys() or "labels_r" in batch.keys() or 'labels_a' in batch.keys():
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


class Linear_ava(nn.Module):
    def __init__(self, config):
        super(Linear_ava, self).__init__()
        self.config = config.MODEL

        if self.config.NUM_PVP == 93:
            self.pvp_matrix = np.array(sentence_pvp93, dtype="float32")
            self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (93,2304)
        elif self.config.NUM_PVP == 76:
            self.pvp_matrix = np.array(sentence_only, dtype="float32")
            self.pvp_matrix = torch.from_numpy(self.pvp_matrix).cuda()  # (93,2304)

        self.probability = nn.Sigmoid()

        from AVA_utils import verb80_weight
        verb80_weight = torch.from_numpy(np.array(verb80_weight)).cuda()
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=verb80_weight)

        self.linear = nn.Linear(
            self.config.NUM_PVP * self.config.BERT_DIM, self.config.NUM_CLASS
        )

    def forward(self, batch):
        pvp_L = torch.einsum("ij,jk->ijk", batch["gt_pvp"], self.pvp_matrix)
        pvp_L = pvp_L.flatten(1, 2)

        s = self.linear(pvp_L)

        p = self.probability(s)

        output = {}
        output["s"] = s
        output["p"] = p

        if "gt_label" in batch.keys():
            output["L_cls"] = self.classification_loss(s, batch["gt_label"])
            output["loss"] = output["L_cls"]
        if "labels_a" in batch.keys():
            output["L_cls"] = self.classification_loss(s, batch["labels_a"])
            output["loss"] = output["L_cls"]

        return output


class Linear_ava_10v(nn.Module):
    def __init__(self, config):
        super(Linear_ava_10v, self).__init__()
        self.config = config.MODEL

        self.probability = nn.Sigmoid()
        from AVA_utils import verb80_weight
        verb80_weight = torch.from_numpy(np.array(verb80_weight)).cuda()
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=verb80_weight)

        self.linear = nn.Linear(
            10 * self.config.VISUAL_DIM, self.config.NUM_CLASS
        )

    def forward(self, batch):
        self.part_V = torch.cat(
            [batch[key].unsqueeze(1)
             for key in
             ['FP0l', 'FP0r', 'FP1l', 'FP1r', 'FP2', 'FP3l', 'FP3r', 'FP4l', 'FP4r', 'FP5']],
            dim=1)  # (bz,10,P_dim)

        self.part_V = batch['gt_part'].unsqueeze(-1).expand(
            self.part_V.shape) * self.part_V  # (bz,10)*(bz,10,P_dim)=(bz,10,P_dim)

        # for F in ['FO', 'FH']:
        #     if F in self.config.F_keys:
        #         self.part_V = torch.cat(
        #             [self.part_V, batch[F].unsqueeze(1).expand(self.part_V.shape)],
        #             dim=-1)  # (bz,10,2048)


        s = self.linear(self.part_V.flatten(1, 2))

        p = self.probability(s)

        output = {}
        output["s"] = s
        output["p"] = p

        if "labels_a" in batch.keys():
            output["L_cls"] = self.classification_loss(s, batch["labels_a"])
            output["loss"] = output["L_cls"]

        return output
