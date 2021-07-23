import os
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import torch
from torch.nn import functional
import pandas as pd
import numpy as np
import sys
import random

# 数据说明
# All node index starts from 1. The zero index is reserved for null during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.
# 数据读取
DATA = 'OAG'
# mathoverflow节点数过多，GAE和VGAE不能使用邻接矩阵，超内存，所以要去掉其中一些节点
# ['socialevolve_2weeks', 'enron', 'CollegeMsg', 'mathoverflow', 'bitcoinotc']
# Load data and sanity check
# graph_data = {
#    ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
#    ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
#    ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))
# }
# g = dgl.heterograph(graph_data)
# print(g.nodes('drug'))
# exit()
# dgl的节点id是从0开始的，不会给你重新设置id
dir_path = '../dataset_{}'.format(DATA)
graph_data = {}
with open(os.path.join(dir_path, 'author_affiliation.txt'), 'r') as fr:
    lines = fr.readlines()
    author_ids = []
    affiliation_ids = []
    # 原本的author_id到图中id的映射
    author2graph = {}
    affiliation2graph = {}
    for line in lines:
        author_id, affiliation_id, time_id = map(int, line.strip('\n').split('\t'))
        if author_id not in author2graph:
            author2graph[author_id] = len(author2graph)
        if affiliation_id not in affiliation2graph:
            affiliation2graph[affiliation_id] = len(affiliation2graph)
        author_ids.append(author2graph[author_id])
        affiliation_ids.append(affiliation2graph[affiliation_id])
    graph_data[('author', 'in', 'affiliation')] = (torch.tensor(author_ids), torch.tensor(affiliation_ids))
with open(os.path.join(dir_path, 'author_field.txt'), 'r') as fr:
    lines = fr.readlines()
    author_ids = []
    field_ids = []
    # 原本的author_id到图中id的映射
    author2graph = {}
    field2graph = {}
    for line in lines:
        author_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
        if author_id not in author2graph:
            author2graph[author_id] = len(author2graph)
        if field_id not in field2graph:
            field2graph[field_id] = len(field2graph)
        author_ids.append(author2graph[author_id])
        field_ids.append(field2graph[field_id])
    graph_data[('author', 'study', 'field')] = (torch.tensor(author_ids), torch.tensor(field_ids))
with open(os.path.join(dir_path, 'author_venue.txt'), 'r') as fr:
    lines = fr.readlines()
    author_ids = []
    venue_ids = []
    # 原本的author_id到图中id的映射
    author2graph = {}
    venue2graph = {}
    for line in lines:
        author_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
        if author_id not in author2graph:
            author2graph[author_id] = len(author2graph)
        if venue_id not in venue2graph:
            venue2graph[venue_id] = len(venue2graph)
        author_ids.append(author2graph[author_id])
        venue_ids.append(venue2graph[venue_id])
    graph_data[('author', 'contribute', 'venue')] = (torch.tensor(author_ids), torch.tensor(venue_ids))

g = dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)
exit()
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
assert (min(src_l.min(), dst_l.min()) == 1)
# assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1) # the nodes need to map one-to-one to the node feat matrix
print(len(n_feat)-1, len(e_feat)-1)
sys.exit()
# print(src_l.shape)
# for i in range(len(src_l)):
#     if src_l[i] == 103 or dst_l[i] == 103:
#         print(src_l[i], dst_l[i], e_idx_l[i], ts_l[i])
#
# import sys
# sys.exit()
# print(torch.tensor(n_feat).shape)
# print(torch.eye(n_feat.shape[0]).shape)
# print(e_feat[1])

# print(torch.eye(10))

# 选择k个时间节点
k = 1
choose_rate = 1# 0.1 matho用的比例
span = max(ts_l)-min(ts_l)
cell = span/k
w = span/k/2
t_sampled = [min(ts_l)+cell/2+cell*i for i in range(k)]
# print(min(ts_l), max(ts_l), t_sampled)
# print(e_feat.shape[0])

# new_feat = torch.tensor()
# 2显示的是索引
# for i in range(len(n_feat)):
#     idx = torch.tensor([i])
#     label2one_hot = functional.one_hot(idx, num_classes=len(n_feat))
#     new_feat = torch.cat(new_feat, label2one_hot)
# n_feat = torch.from_numpy(np.array(new_feat))


# 生成k个数据集
for i in range(k):

    node_num = int(len(n_feat)*choose_rate)
    choose_idx = random.sample(range(1, len(n_feat) + 1), node_num)
    # print(delete_idx)
    # sys.exit()
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    src, dst, e_idx, label, ts, e_f = [0], [0], [0], [0], [0], [e_feat[0]]
    # 静态图中没有时间戳，所以要判断当前边是否出现
    edge_set = {'{}-{}'.format(0, 0)}
    for j in range(len(src_l)):
        # if src_l[j] not in choose_idx or dst_l[j] not in choose_idx:
        if src_l[j] >= node_num or dst_l[j] >= node_num:
            continue
        if abs(ts_l[j] - t_sampled[i]) <= w and '{}-{}'.format(src_l[j], dst_l[j]) not in edge_set:
        # mathoverflow要判断是否删除节点，而且ts_l正好是顺序的，就节省点时间
        # if '{}-{}'.format(src_l[j], dst_l[j]) not in edge_set:
            src.append(src_l[j])
            dst.append(dst_l[j])
            e_idx.append(e_idx_l[j])
            label.append(label_l[j])
            ts.append(ts_l[j])
            e_f.append(e_feat[e_idx_l[j]])
            # 去掉重复出现的边
            edge_set.add('{}-{}'.format(src_l[j], dst_l[j]))

    # 只考虑有向图
    g.add_edges(src, dst)
    # g.edata['idx'] = torch.tensor(e_idx)
    # g.edata['ts'] = torch.tensor(ts)
    # g.edata['feat'] = torch.tensor(e_f)
    # g.ndata['feat'] = torch.tensor(n_feat).type(torch.float32)
    # 由于wikipedia的节点向量都为0，所以需要进行初始化
    # print(n_feat.shape)
    # sys.exit()
    g.ndata['feat'] = torch.rand((node_num, 16))/100
    # print(g.ndata['feat'][:2])
    # sys.exit()
    g.ndata['label'] = torch.tensor(torch.zeros(node_num))
    # g.ndata['feature'] = torch.eye(n_feat.shape[0])

    # print(g)
    # print(g.ndata['feature'][1])


    graph_path = '{}-{}-{}.dgl'.format(DATA, i, k)
    save_graphs(graph_path, g)#, {'idx': g.edata['idx']})
    print(g)