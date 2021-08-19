"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GATConv


class GAT_NC(nn.Module):
    def __init__(self,
                 g,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_NC, self).__init__()
        self.g = g
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l-1], hidden_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # hidden vector projection
        # self.gat_layers.append(GATConv(
        #     hidden_dim * heads[-2], hidden_dim, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        # output vector projection
        self.gat_out = GATConv(
            hidden_dim * heads[-1], out_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None)
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, inputs):
        h = inputs
        for l in range(self.n_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_out(self.g, h).mean(1)
        # h = self.gat_layers[-1](self.g, h).flatten(1)
        seq_fts = self.fc(inputs)
        return logits, h, seq_fts

class GAT_LP(nn.Module):
    def __init__(self,
                 g,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_LP, self).__init__()
        self.g = g
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l-1], hidden_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True))
        # output projection
        # self.gat_layers.append(GATConv(
        #     hidden_dim * heads[-2], hidden_dim, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True))
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, inputs):
        h = inputs
        for l in range(self.n_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        adj_rec = torch.sigmoid(torch.matmul(h, h.t()))
        # 将特征进行线性变换，用来进行对比学习
        seq_fts = self.fc(inputs)
        return adj_rec, h, seq_fts