import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *

from .model import BaseModel


class IMF(BaseModel):
    def __init__(self, args):
        super(IMF, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)
        if args.dataset == "DB15K":
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            # multi-modal information for MKG
            img = args.img.to(self.device)[:, 0: args.dim].view(args.img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
            txt = txt.view(txt.size(0), -1)
        elif "Kuai" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "TIVA" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 32, 64))
            img = img.view(img.size(0), -1)
            txt = args.desp.to(self.device)[:, 0: 256]
            txt = txt.view(txt.size(0), -1)
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()
        
    def contrastive_loss(self, s_embed, v_embed, t_embed):
        s_embed, v_embed, t_embed = s_embed / torch.norm(s_embed), v_embed / torch.norm(v_embed), t_embed / torch.norm(t_embed)
        pos_sv = torch.sum(s_embed * v_embed, dim=1, keepdim=True)
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)
        neg_s = torch.matmul(s_embed, s_embed.t())
        neg_v = torch.matmul(v_embed, v_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_s = neg_s - torch.diag_embed(torch.diag(neg_s))
        neg_v = neg_v - torch.diag_embed(torch.diag(neg_v))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(torch.cat([pos_sv, pos_st, pos_vt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_v, neg_t], dim=1), dim=1)
        loss = torch.mean(F.softplus(neg - pos))
        return loss
        
    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_img_embed = self.img_entity_embeddings(head)
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed = self.txt_entity_embeddings(head)
        r_txt_embed = self.txt_relation_embeddings(relation)
        e_mm_embed = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)
        r_mm_embed = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)

        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_i = torch.mm(pred_i, self.img_entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.img_entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight).transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)
        return [pred_s, pred_i, pred_d, pred_mm]

    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        return loss_s + loss_i + loss_d + loss_mm


    def loss_con(self, batch_inputs):
        head = batch_inputs[:, 0]
        e_embed = self.entity_embeddings(head)
        e_img_embed = self.img_entity_embeddings(head)
        e_txt_embed = self.txt_entity_embeddings(head)
        loss_con = self.contrastive_loss(e_embed, e_img_embed, e_txt_embed)
        return loss_con