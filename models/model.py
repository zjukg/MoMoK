import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "MR": 100000, "MRR": -1}


class Mutan(BaseModel):
    def __init__(self, args):
        super(Mutan, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.Mutan = MutanLayer(args.dim, 5)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.Mutan(e_embed, r_embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class TuckER(BaseModel):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
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
        self.dim = args.dim
        self.TuckER = TuckERLayer(args.dim, args.r_dim)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs, lookup=None):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.TuckER(e_embed, r_embed)
        if lookup is None:
            pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        else:
            pred = torch.bmm(pred.unsqueeze(1), self.entity_embeddings.weight[lookup].transpose(1, 2)).squeeze(1)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class ConvE(BaseModel):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.ConvE = ConvELayer(args.dim, args.out_channels, args.kernel_size, args.k_h, args.k_w)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_embed = e_embed.view(-1, 1, self.dim)
        r_embed = r_embed.view(-1, 1, self.dim)
        embed = torch.cat([e_embed, r_embed], dim=1)
        embed = torch.transpose(embed, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))

        pred = self.ConvE(embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)



