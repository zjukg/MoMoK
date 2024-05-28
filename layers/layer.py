import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()


class ConvKBLayer(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super(ConvKBLayer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x


class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout2d(0.2)
        self.hidden_drop = nn.Dropout(0.2)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        assert k_h * k_w == dim
        flat_sz_h = int(2*k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if(CUDA):
                edge_sources = edge_sources.to('cuda:0')
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge):
        N = input.size()[0]

        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        edge_m = self.W.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_m).squeeze())).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(nfeat, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, entity_embeddings, relation_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x_rel = relation_embeddings.mm(self.W)
        x = F.elu(self.out_att(x, edge_list))
        return x, x_rel


class ModalFusionLayerForThreeModal(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayerForThreeModal, self).__init__()

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