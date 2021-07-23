import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model_HGT import *
import os
from dgl.data.utils import save_graphs, load_graphs
import dgl.function as fn
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from data_process import *

# print(type(data['PvsA']))
# print(data['PvsA'][0])
# exit()
# 0.5及以后的版本用不了现在的数据读取方法，且高版本不能直接输出图G
# https://blog.csdn.net/ShakalakaPHD/article/details/114526374




# 划分训练集，两种办法，1、edge_subgraph，2、给边和节点打上标签
# G.edges['contribute'].data['train_mask'] = torch.zeros(G.number_of_edges('contribute'), dtype=torch.bool).bernoulli(0.7)
# G.edges['contribute'].data['valid_mask'] = torch.zeros(G.number_of_edges('contirbute'), dtype=torch.bool).bernoulli(0.7)


# create train graph
# train_edge_idx = torch.tensor(train_edge_idx).to(device)
# dgl.graph()



# adj = train_graph.adjacency_matrix(etype='writing').to_dense().to(device)

# compute loss parameters
# weight_tensor, norm = compute_loss_para(adj)

# device = torch.device("cuda:0")


# model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item() + 1, n_layers=2, n_heads=4, use_norm=True).to(device)

if __name__ == '__main__':
    data_file_path = 'ACM.mat'
    data = scipy.io.loadmat(data_file_path)
    device = torch.device("cuda:0")

    # 异质图必须明确节点和边的类型，否则会报错
    G = read_data()
    print(G)

    # 确定训练集、验证集和测试集
    # 这里的邻接矩阵作用只是提供idx的范围
    adj_orig = G.adjacency_matrix(etype='written-by').to_dense()
    # print(adj_orig.shape)
    # exit()
    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(G, adj_orig)
    train_edge_idx = torch.tensor(train_edge_idx)

    # 老版本preserve_nodes=True会报错
    train_graph = G.edge_subgraph({
        # 用writing会报错，模型部分author的节点没有设置't'属性，猜测是multi_update_all函数只更新尾实体，所以author没有添加't'属性
        ('paper', 'written-by', 'author'): train_edge_idx,  # list(range(G.number_of_edges('written-by'))),
        # ('author', 'writing', 'paper'): train_edge_idx,#list(range(G.number_of_edges('writing'))),
        ('paper', 'citing', 'paper'): list(range(G.number_of_edges('citing'))),
        ('paper', 'cited', 'paper'): list(range(G.number_of_edges('cited'))),
        ('paper', 'is-about', 'subject'): list(range(G.number_of_edges('is-about'))),
        ('subject', 'has', 'paper'): list(range(G.number_of_edges('has'))),
        ('paper', 'contribute', 'venue'): list(range(G.number_of_edges('contribute'))),
        ('venue', 'has-paper', 'paper'): list(range(G.number_of_edges('has-paper'))),
    }, preserve_nodes=True)
    print(train_graph)
    G = train_graph

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
        negative_graph = construct_negative_graph(G, 1, ('paper', 'written-by', 'author'))
        # node_features = {}
        # for ntype in G.ntypes:
        #     node_features[ntype] = G.nodes[ntype].data['feature']
        pos_score, neg_score = model(G, negative_graph, ('paper', 'written-by', 'author'))
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step()

        train_acc = get_acc(pos_score, neg_score)
        if train_step % 10 == 0:
            val_roc, val_ap = get_score(G, val_edges, val_edges_false, ('paper', 'written-by', 'author'))
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
                  "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))
            continue
            # print('Valid Accuracy:', get_score())
        print("Epoch:", '%04d' % (epoch + 1), 'Loss:', loss.item(), "Train Accuracy:", train_acc)

    test_roc, test_ap = get_score(G, test_edges, test_edges_false, ('paper', 'written-by', 'author'))
    print("val_roc=", "{:.5f}".format(test_roc), "val_ap=", "{:.5f}".format(test_ap))