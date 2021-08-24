"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl

class GCN_NC(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_NC, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        # output layer
        self.output = GraphConv(hidden_dim, out_dim)
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        logits = self.output(self.g, h)
        seq_fts = self.fc(features)
        return logits, h, seq_fts

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            # 计算边的两个节点的点积，作为这条边的score属性，这里没有经过sigmoid
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
            # graph.edges[etype].data['score'] = torch.sigmoid(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']

class GCN_LP(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_LP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation,allow_zero_in_degree=True, weight=True))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation, allow_zero_in_degree=True,
                                         weight=True))
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

        adj_rec = torch.sigmoid(torch.matmul(h, h.t()))
        # 将特征进行线性变换，用来进行对比学习
        seq_fts = self.fc(features)
        # GCN模型的问题，相邻节点会逐渐趋同，所以加上线性变换，让相邻节点有所不同
        h = h+seq_fts
        h = self.norm(h)
        return adj_rec, h, seq_fts