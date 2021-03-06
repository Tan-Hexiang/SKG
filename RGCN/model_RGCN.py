"""RGCN layer implementation"""
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import tqdm

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)
        # print(hs)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 g,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed
            # ?????????????????????
            # self.embeds[ntype] = nn.Parameter(th.Tensor(g.nodes[ntype].data['feature'].to('cpu')))

        # print(self.embeds)

    def forward(self, block=None):
        """Forward computation
        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.
        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds

class RGCN_NC(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(RGCN_NC, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        # if num_bases < 0 or num_bases > len(self.rel_names):
        #     self.num_bases = len(self.rel_names)
        # else:
        #     self.num_bases = num_bases
        # ???????????????????????????????????????
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.in_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.hidden_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.hidden_dim, self.hidden_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.out_layer = RelGraphConvLayer(
            self.hidden_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop)
        # self.layers.append(RelGraphConvLayer(
        #     self.hidden_dim, self.out_dim, self.rel_names,
        #     self.num_bases, activation=None,
        #     self_loop=self.use_self_loop))
        # ????????????
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.in_dim, self.hidden_dim, bias=False)

    def forward(self, h=None):
        if h is None:
            # full graph training
            h = self.embed_layer()
        # full graph training
        for layer in self.layers:
            h = layer(self.g, h)
        output = self.out_layer(self.g, h)

        for ntype in self.g.ntypes:
            self.g.nodes[ntype].data['h'] = self.out(h[ntype])
            self.g.nodes[ntype].data['w'] = self.fc(self.embed_layer()[ntype])
        return output

    def inference(self, g, batch_size, device, num_workers, x=None):
        """Minibatch inference of final representation over all node types.
        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.
        """

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: th.zeros(
                    g.number_of_nodes(k),
                    self.hidden_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: th.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h??????5.1???????????????????????????????????????????????????????????????
        with graph.local_scope():
            graph.ndata['h'] = h
            # ??????????????????????????????????????????????????????score???????????????????????????sigmoid
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
            # graph.edges[etype].data['score'] = torch.sigmoid(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']

class RGCN_LP(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(RGCN_LP, self).__init__()
        self.g = g
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        # num_bases???????????????????????????????????????????????????-1?????????????????????????????????????????????
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.hidden_dim)
        self.layers = nn.ModuleList()
        # i2h
        # ??????????????????????????????in_dim??????hidden_dim??????????????????????????????????????????
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.hidden_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))
        # h2h
        for i in range(self.num_hidden_layers-1):
            self.layers.append(RelGraphConvLayer(
                self.hidden_dim, self.hidden_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # ??????????????????????????????????????????????????????
        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.pred = HeteroDotProductPredictor()
        # ?????????SN?????????????????????????????????
        self.fc = nn.Linear(self.hidden_dim, self.out_dim, bias=False)

    def forward(self, neg_G, etype):
        h = self.embed_layer()
        for layer in self.layers:
            h = layer(self.g, h)
        # h??????????????????????????????????????????????????????
        # print(h)
        for ntype in self.g.ntypes:
            self.g.nodes[ntype].data['h'] = self.out(h[ntype])
            self.g.nodes[ntype].data['w'] = self.fc(self.embed_layer()[ntype])
        h_dict = {ntype: self.g.nodes[ntype].data['h'] for ntype in self.g.ntypes}
        return self.pred(self.g, h_dict, etype), self.pred(neg_G, h_dict, etype)