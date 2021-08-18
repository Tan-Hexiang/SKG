import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            # print(ntype)
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
                
class HGT_NC(nn.Module):
    def __init__(self, G, in_dim, hidden_dim, out_dim, n_layers, n_heads, use_norm = True):
        super(HGT_NC, self).__init__()
        self.g = G
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, len(G.node_dict), len(G.edge_dict), n_heads, use_norm = use_norm))
        self.out = nn.Linear(hidden_dim, out_dim)
        # 相当于SN中对每个节点的线性变换
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self):
        # out_key表示对哪类节点做分类
        # 最终返回每个节点是每类标签的概率
        for ntype in self.g.ntypes:
            n_id = self.g.node_dict[ntype]
            self.g.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](self.g.nodes[ntype].data['feature']))
        for i in range(self.n_layers):
            self.gcs[i](self.g, 'h', 'h')
        # 输出所有类型的节点的分类结果，其中只有一类节点是有效分类
        output = {}
        for ntype in self.g.ntypes:
            output[ntype] = self.out(self.g.nodes[ntype].data['h'])
            self.g.nodes[ntype].data['w'] = self.fc(self.g.nodes[ntype].data['feature'])
        return output
    def __repr__(self):
        return '{}(in_dim={}, hidden_dim={}, out_dim={}, n_layers={})'.format(
            self.__class__.__name__, self.in_dim, self.hidden_dim,
            self.out_dim, self.n_layers)

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            # 计算边的两个节点的点积，作为这条边的score属性，这里没有经过sigmoid
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
            # graph.edges[etype].data['score'] = torch.sigmoid(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']

class HGT_LP(nn.Module):
    def __init__(self, G, in_dim, hidden_dim, out_dim, n_layers, n_heads, use_norm = True):
        super(HGT_LP, self).__init__()
        self.g = G
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        # 每类节点做映射
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, len(G.node_dict), len(G.edge_dict), n_heads, use_norm = use_norm))
        # 这里可以给每类节点一个单独的映射矩阵
        self.out = nn.Linear(hidden_dim, out_dim)
        self.pred = HeteroDotProductPredictor()
        # 相当于SN中对每个节点的线性变换
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, nega_G, etype):
        # etype表示对哪类边做分类
        G = self.g
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            # 'h'表示隐状态向量，n_id表示使用第几个映射矩阵
            G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['feature']))
        for i in range(self.n_layers):
            # 两个'h'分别表示输入特征和输出特征是节点的哪个属性
            self.gcs[i](G, 'h', 'h')
        for ntype in G.ntypes:
            G.nodes[ntype].data['h'] = self.out(G.nodes[ntype].data['h'])
            G.nodes[ntype].data['w'] = self.fc(G.nodes[ntype].data['feature'])
        h = {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}
        return self.pred(G, h, etype), self.pred(nega_G, h, etype)

    def __repr__(self):
        return '{}(in_dim={}, hidden_dim={}, out_dim={}, n_layers={})'.format(
            self.__class__.__name__, self.in_dim, self.hidden_dim,
            self.out_dim, self.n_layers)
