import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import os
import torch
from dgl.data.utils import save_graphs, load_graphs
import dgl.function as fn



# print(type(data['PvsA']))
# print(data['PvsA'][0])
# exit()
# 0.5及以后的版本用不了现在的数据读取方法，且高版本不能直接输出图G
# https://blog.csdn.net/ShakalakaPHD/article/details/114526374

def get_graph(DATA='OAG'):
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
    return g


# 构建异质图，这里把无向图变成双向图
# print(type(data['PvsA']))
# exit()
# for i in data['PvsA']:
#     print(type(i))
#     exit()
def write_data():
    matrix = data['PvsA'].toarray()
    print(matrix.shape, data['PvsA'].shape)
    with open('../dataset_ACM/paper-author.txt', 'w') as fw:
        for i in range(data['PvsA'].shape[0]):
            for j in range(data['PvsA'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsP'].toarray()
    print(matrix.shape, data['PvsP'].shape)
    with open('../dataset_ACM/paper-paper.txt', 'w') as fw:
        for i in range(data['PvsP'].shape[0]):
            for j in range(data['PvsP'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsL'].toarray()
    print(matrix.shape, data['PvsL'].shape)
    with open('../dataset_ACM/paper-field.txt', 'w') as fw:
        for i in range(data['PvsL'].shape[0]):
            for j in range(data['PvsL'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsC'].toarray()
    print(matrix.shape, data['PvsC'].shape)
    with open('../dataset_ACM/paper-venue.txt', 'w') as fw:
        for i in range(data['PvsC'].shape[0]):
            for j in range(data['PvsC'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')

def read_data():
    dir_path = '../dataset_ACM'
    graph_data = {}
    with open(os.path.join(dir_path, 'paper-author.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        author_ids = []
        for line in lines:
            paper_id, author_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            author_ids.append(author_id)
        graph_data[('paper', 'written-by', 'author')] = (torch.tensor(paper_ids), torch.tensor(author_ids))
        graph_data[('author', 'writing', 'paper')] = (torch.tensor(author_ids), torch.tensor(paper_ids))
    with open(os.path.join(dir_path, 'paper-paper.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids1 = []
        paper_ids2 = []
        for line in lines:
            paper_id1, paper_id2, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids1.append(paper_id1)
            paper_ids2.append(paper_id2)
        graph_data[('paper', 'citing', 'paper')] = (torch.tensor(paper_ids1), torch.tensor(paper_ids2))
        graph_data[('paper', 'cited', 'paper')] = (torch.tensor(paper_ids2), torch.tensor(paper_ids1))
    with open(os.path.join(dir_path, 'paper-field.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        field_ids = []
        for line in lines:
            paper_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            field_ids.append(field_id)
        graph_data[('paper', 'is-about', 'subject')] = (torch.tensor(paper_ids), torch.tensor(field_ids))
        graph_data[('subject', 'has', 'paper')] = (torch.tensor(field_ids), torch.tensor(paper_ids))
    with open(os.path.join(dir_path, 'paper-venue.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        venue_ids = []
        for line in lines:
            paper_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            venue_ids.append(venue_id)
        graph_data[('paper', 'contribute', 'venue')] = (torch.tensor(paper_ids), torch.tensor(venue_ids))
        # graph_data[('venue', 'has-paper', 'paper')] = (torch.tensor(venue_ids), torch.tensor(paper_ids))

    g = dgl.heterograph(graph_data)
    return g
#
#
# write_data()
# G = dgl.heterograph({
#     ('paper', 'written-by', 'author'): data['PvsA'],
#     ('author', 'writing', 'paper'): data['PvsA'].transpose(),
#     ('paper', 'citing', 'paper'): data['PvsP'],
#     ('paper', 'cited', 'paper'): data['PvsP'].transpose(),
#     ('paper', 'is-about', 'subject'): data['PvsL'],
#     ('subject', 'has', 'paper'): data['PvsL'].transpose(),
#     ('paper', 'contribute', 'venue'): data['PvsC'],
#     # ('venue', 'has-paper', 'paper'): data['PvsC'].transpose(),
# })
# graph_path = '{}-{}-{}.dgl'.format(DATA, i, k)
# graph_path = 'ACM.dgl'
# save_graphs(graph_path, [G])
# G = load_graphs(graph_path)
# G = get_graph()
# G.to(device)
# 异质图必须明确节点和边的类型，否则会报错
# print(G)

# exit()
# print(G.edges(etype='contribute'))
# print(G.number_of_edges('contribute'))
# exit()

# 返回稀疏矩阵的csr_matrix形式
# pvc = data['PvsC'].tocsr()
# 返回稀疏矩阵的coo_matrix形式
# p_selected = pvc.tocoo()

# generate labels
# 这是一个节点分类任务
# labels = pvc.indices
# labels = torch.tensor(labels).long()
# print(labels.max().item())
# exit()

# generate train/val/test split
# pid = p_selected.row
# shuffle = np.random.permutation(pid)
# train_idx = torch.tensor(shuffle[0:800]).long()
# val_idx = torch.tensor(shuffle[800:900]).long()
# test_idx = torch.tensor(shuffle[900:]).long()


def mask_test_edges_dgl(graph, adj):
    src, dst = graph.edges(etype='contribute')
    # src是paper，dst是venue
    # adj的行数是venue，列数是paper
    # print(src, dst)
    # exit()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    # 验证集0.2，测试集0.1
    num_test = int(np.floor(edges_all.shape[0] * 0.1))
    num_val = int(np.floor(edges_all.shape[0] * 0.2))

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = all_edge_idx[(num_val + num_test):]
    test_edges = edges_all[test_edge_idx]
    val_edges = edges_all[val_edge_idx]
    train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[1])
        idx_j = np.random.randint(0, adj.shape[0])
        # if idx_i == idx_j:
        #     continue
        # 判断负样本是否出现过
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            # 这里是有向图
            # if ismember([idx_j, idx_i], np.array(test_edges_false)):
            #     continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[1])
        idx_j = np.random.randint(0, adj.shape[0])
        # if idx_i == idx_j:
        #     continue
        # 验证集的负样本只要不在训练集和验证集中出现就可以
        if ismember([idx_i, idx_j], train_edges):
            continue
        # if ismember([idx_j, idx_i], train_edges):
        #     continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        if val_edges_false:
            # if ismember([idx_j, idx_i], np.array(val_edges_false)):
            #     continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))#.to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def construct_negative_graph(graph, k, etype):
    # k表示负样本的比例
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

    # 交叉熵
    # 先把分数转成概率
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(neg_score)
    label = torch.cat((torch.ones(pos_score.shape), torch.zeros(neg_score.shape)), 0).to(device)
    pred = torch.cat((pos_score, neg_score), 0).to(device)
    loss = F.binary_cross_entropy(pred.view(-1), label.view(-1))
    return loss

def get_acc(pos_score, neg_score):
    # pos_score>0.5的是预测正确的，neg_score<0.5的是正确的
    accuracy = (torch.sigmoid(pos_score)>0.5).sum().float()+(torch.sigmoid(neg_score)<0.5).sum().float()
    accuracy = accuracy / (pos_score.shape[0]+neg_score.shape[0])
    return float(accuracy)

# 确定训练集、验证集和测试集
# 这里的邻接矩阵作用只是提供idx的范围
# adj_orig = G.adjacency_matrix(etype='contribute').to_dense()
# print(adj_orig.shape)
# exit()
# build test set with 10% positive links
# train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(G, adj_orig)
# train_edge_idx = torch.tensor(train_edge_idx)#.to(device)
# 划分训练集，两种办法，1、edge_subgraph，2、给边和节点打上标签
# G.edges['contribute'].data['train_mask'] = torch.zeros(G.number_of_edges('contribute'), dtype=torch.bool).bernoulli(0.7)
# G.edges['contribute'].data['valid_mask'] = torch.zeros(G.number_of_edges('contirbute'), dtype=torch.bool).bernoulli(0.7)


# create train graph
# train_edge_idx = torch.tensor(train_edge_idx).to(device)
# dgl.graph()
# 这个版本preserve_nodes=True会报错，不知道为什么
# train_graph = G.edge_subgraph({
#     # ('paper', 'written-by', 'author'): list(range(G.number_of_edges('written-by'))),
#     ('author', 'writing', 'paper'): train_edge_idx,#list(range(G.number_of_edges('writing'))),
#     ('paper', 'citing', 'paper'): list(range(G.number_of_edges('citing'))),
#     ('paper', 'cited', 'paper'): list(range(G.number_of_edges('cited'))),
#     ('paper', 'is-about', 'subject'): list(range(G.number_of_edges('is-about'))),
#     ('subject', 'has', 'paper'): list(range(G.number_of_edges('has'))),
#     ('paper', 'contribute', 'venue'): list(range(G.number_of_edges('contribute'))),
#     ('venue', 'has-paper', 'paper'): list(range(G.number_of_edges('has-paper')))
# }, preserve_nodes=True)
# print(train_graph)

# train_graph = train_graph.to(device)
# adj = train_graph.adjacency_matrix(etype='contribute').to_dense().to(device)

# compute loss parameters
# weight_tensor, norm = compute_loss_para(adj)

# device = torch.device("cuda:0")


# model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item() + 1, n_layers=2, n_heads=4, use_norm=True).to(device)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_file_path = 'ACM.mat'
    data = scipy.io.loadmat(data_file_path)

    G = read_data()
    print(G)

    G.node_dict = {}
    G.edge_dict = {}
    # 给每个类型加上id，从0开始
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

    # 随机生成每个节点的向量
    node_features = {}
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad=False)  # .to(device)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['feature'] = emb
        node_features[ntype] = emb

    model = HGT_PF(G, 400, 200, 50, n_layers=2, n_heads=4, use_norm=True)  # .to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=1000, max_lr=1e-3, pct_start=0.05)

    best_val_acc = 0
    best_test_acc = 0
    train_step = 0
    for epoch in range(1000):
        # 这里是在整张图上做训练
        model.train()
        negative_graph = construct_negative_graph(G, 1, ('paper', 'contribute', 'venue'))
        node_features = {}
        for ntype in G.ntypes:
            node_features[ntype] = G.nodes[ntype].data['feature']
        pos_score, neg_score = model(G, negative_graph, ('paper', 'contribute', 'venue'))
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        print('Loss:', loss.item(), "Accuracy:", get_acc(pos_score, neg_score))