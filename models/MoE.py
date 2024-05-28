import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
from .modules import ContrastiveLoss

from .model import BaseModel


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)
    


import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
from .modules import ContrastiveLoss

from .model import BaseModel

class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        # x_mm = torch.relu(x_mm)
        return x_mm
    
    def relation_gated_fuse(self, modal1_emb, modal2_emb, modal3_emb, rel):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores / rel, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.mean(1).view(batch_size, self.out_dim)
        x_mm = torch.relu(x_mm)
        return x_mm
    
    def gated_fusion(self, emb, rel):
        # emb: batch_size x dim
        # rel: batch_size x dim
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel



class KoMoE(BaseModel):
    def __init__(self, args):
        super(KoMoE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id), 
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)
        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)

        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        # Score Functions
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.fuse_out_dim = self.dim
        # Score function layers
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.img_dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.txt_dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, self.fuse_out_dim)
        # Multi-modal fusion layers

        self.visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim])
        self.text_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim])
        self.structure_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.dim, self.dim])
        self.mm_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim])
    
        self.fuse_e = ModalFusionLayer(
            in_dim=args.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.img_dim,
            txt_dim=self.txt_dim
        )
        self.fuse_r = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim
        )
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

        
    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.structure_moe(self.entity_embeddings(head))
        r_embed = self.relation_embeddings(relation)
        e_img_embed = self.visual_moe(self.img_entity_embeddings(head))
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed = self.text_moe(self.txt_entity_embeddings(head))
        r_txt_embed = self.txt_relation_embeddings(relation)
        rel_gate = self.rel_gate(relation)
        e_mm_embed = self.fuse_e.relation_gated_fuse(e_embed, e_img_embed, e_txt_embed, rel_gate)
        r_mm_embed = self.fuse_r.relation_gated_fuse(r_embed, r_img_embed, r_txt_embed, rel_gate)
        e_mm_embed = self.mm_moe(e_mm_embed)

        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        
        all_s = self.structure_moe(self.entity_embeddings.weight)
        all_v = self.visual_moe(self.img_entity_embeddings.weight)
        all_t = self.text_moe(self.txt_entity_embeddings.weight)
        all_f = self.fuse_e(all_s, all_v, all_t)
        """
        # 第二版，Answer不做MoE
        all_s = self.entity_embeddings.weight
        all_v = self.img_entity_embeddings.weight
        all_t = self.txt_entity_embeddings.weight
        all_f = self.fuse_e(all_s, all_v, all_t)
        """
        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))

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
