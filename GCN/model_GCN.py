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

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            # 计算边的两个节点的点积，作为这条边的score属性，这里没有经过sigmoid
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
            # graph.edges[etype].data['score'] = torch.sigmoid(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']

class GCN_PF(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_PF, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation,allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        adj_rec = torch.sigmoid(torch.matmul(h, h.t()))
        return adj_rec, h