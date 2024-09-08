import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


def full_attention_conv(qs, ks, vs, output_attn=False):
    '''
    qs: query tensor [N, H, M] # M = D
    ks: key tensor [L, H, M] # L = N since MLP(in, out=in)
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    # normalize input
    qs = qs / torch.norm(qs, p=2) # [N, H, M]
    ks = ks / torch.norm(ks, p=2) # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
    all_ones = torch.ones([vs.shape[0]]).to(vs.device)
    vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
    attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer.squeeze(2) # [N, L, H] / [N, H]
    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, edge_index, edge_weight):
    # print("Applying gcn_conv! \n")
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

class scGraphformerConv(nn.Module):
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               use_graph=False,
               use_weight=True):
        super(scGraphformerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.use_graph = use_graph

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query,key,value) # [N, H, D]

        if self.use_graph:
            final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output

        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output

class scGraphformer(nn.Module):
    '''
    scGraphformer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=False):
        super(scGraphformer, self).__init__()
        hidden_channels = in_channels
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                scGraphformerConv(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight, use_graph=use_graph))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, 256))
        self.bns.append(nn.BatchNorm1d(256))
        self.fcs.append(nn.Linear(256, out_channels))
        
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        # output MLP layer
        x = self.fcs[-2](x)
        if self.use_bn:
            x = self.bns[-1](x)
        x = self.activation(x)
        x_out = self.fcs[-1](x)
        

        return x_out

    def get_attentions(self, x, edge_index, edge_weight=None):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, edge_index, edge_weight, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0) # [layer num, N, N]