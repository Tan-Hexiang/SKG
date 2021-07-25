# 用HGT做个实验，两个HGT都对异质图做操作，一个异质图只有论文和作者，一个异质图只有论文、会议和领域
import sys
sys.path.append('../HGT-DGL')
sys.path.append('../GAE')
sys.path.append('../GMI')
# from train_paper_venue import *
import data_process_KG
import data_process_SN
import preprocess
from warnings import filterwarnings
import time
import train_GAE
import model_GAE
from model_HGT import *
from model import *
# import dill
import torch
import numpy as np

filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='SKG')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--dim_init', '-h_in', type=int, default=400, help='Dim of initial embedding vector.')
parser.add_argument('--dim_embed', '-h_out', type=int, default=16, help='Dim of final embedding vector.')
parser.add_argument('--neg_num', type=int, default=2, help='Number of negtive sampling of each node.')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--cuda', type=int, default=3, help='GPU id to use.')
args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
# device = torch.device('cpu')

def KG_data_prepare():
    KG = data_process_KG.read_data()
    print(KG)
    # print(KG.adjacency_matrix(etype='written-by'))
    # exit()
    triplet = ('paper', 'written-by', 'author')
    # 这里的邻接矩阵作用只是提供idx的范围，行数是源节点，列数是目标节点
    # KG = KG.to(device)
    adj_orig = KG.adjacency_matrix(etype='written-by').to_dense()
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = data_process_KG.mask_test_edges_dgl(KG, adj_orig, triplet[1])
    train_edge_idx = torch.tensor(train_edge_idx)#.to(device)

    # 老版本preserve_nodes=True会报错
    train_graph = KG.edge_subgraph({
        # 用writing会报错，模型部分author的节点没有设置't'属性，猜测是multi_update_all函数只更新尾实体，所以author没有添加't'属性
        ('paper', 'written-by', 'author'): train_edge_idx,  # list(range(G.number_of_edges('written-by'))),
        # ('author', 'writing', 'paper'): train_edge_idx,#list(range(G.number_of_edges('writing'))),
        # ('paper', 'citing', 'paper'): list(range(G.number_of_edges('citing'))),
        # ('paper', 'cited', 'paper'): list(range(G.number_of_edges('cited'))),
        ('paper', 'is-about', 'subject'): list(range(KG.number_of_edges('is-about'))),
        ('subject', 'has', 'paper'): list(range(KG.number_of_edges('has'))),
        ('paper', 'contribute', 'venue'): list(range(KG.number_of_edges('contribute'))),
        ('venue', 'has-paper', 'paper'): list(range(KG.number_of_edges('has-paper'))),
    }, preserve_nodes=True)
    print(train_graph)
    KG = train_graph

    KG.node_dict = {}
    KG.edge_dict = {}
    # 给每个类型加上id，从0开始
    for ntype in KG.ntypes:
        KG.node_dict[ntype] = len(KG.node_dict)
    for etype in KG.etypes:
        KG.edge_dict[etype] = len(KG.edge_dict)
        # 貌似dgl的图在to(device)后就不能进行更改了
        KG.edges[etype].data['id'] = torch.ones(KG.number_of_edges(etype), dtype=torch.long) * KG.edge_dict[etype]

    # 随机生成每个节点的向量
    node_features = {}
    for ntype in KG.ntypes:
        emb = nn.Parameter(torch.Tensor(KG.number_of_nodes(ntype), args.dim_init), requires_grad=False)  # .to(device)
        nn.init.xavier_uniform_(emb)
        KG.nodes[ntype].data['feature'] = emb
        node_features[ntype] = emb

    KG = KG.to(device)
    model_KG = HGT_PF(KG, args.dim_init, args.hidden1, args.dim_embed, n_layers=2, n_heads=4, use_norm=True).to(device)
    # KG_parameters = KG, 400, 200, 16, 2, 4, True
    return KG, model_KG, triplet, val_edges, val_edges_false, test_edges, test_edges_false

    optimizer = torch.optim.AdamW(model_KG.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=1000, max_lr=1e-3, pct_start=0.05)

    train_step = 0
    for epoch in range(1000):
        # 这里是在整张图上做训练
        model_KG.train()
        negative_graph = data_process_KG.construct_negative_graph(KG, 1, triplet, device)
        pos_score, neg_score = model_KG(KG, negative_graph, triplet)
        loss = data_process_KG.compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step()

        train_acc = data_process_KG.get_acc(pos_score, neg_score)
        val_roc, val_ap = data_process_KG.get_score(KG, val_edges, val_edges_false, ('paper', 'written-by', 'author'))
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))
        continue
            # print('Valid Accuracy:', get_score())
        # print("Epoch:", '%04d' % (epoch + 1), 'Loss:', loss.item(), "Train Accuracy:", train_acc)

    test_roc, test_ap = data_process_KG.get_score(KG, test_edges, test_edges_false, ('paper', 'written-by', 'author'))
    print("val_roc=", "{:.5f}".format(test_roc), "val_ap=", "{:.5f}".format(test_ap))

def SN_data_prepare():
    SN = data_process_SN.read_data_SN()
    print(SN)
    # 生成每个节点的特征向量
    emb = nn.Parameter(torch.Tensor(len(SN.nodes()), args.dim_init), requires_grad=False)#.to(device)
    nn.init.xavier_uniform_(emb)
    SN.ndata['feat'] = emb
    feats = SN.ndata.pop('feat').to(device)
    in_dim = feats.shape[-1]

    # generate input
    adj_orig = SN.adjacency_matrix().to_dense()

    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = preprocess.mask_test_edges_dgl(SN, adj_orig)

    SN = SN.to(device)

    # create train SN
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_SN = dgl.edge_subgraph(SN, train_edge_idx, preserve_nodes=True).to(device)
    train_SN = train_SN.to(device)
    adj = train_SN.adjacency_matrix().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = train_GAE.compute_loss_para(adj, device)

    # create model
    gae_model = model_GAE.GAEModel(in_dim, args.hidden1, args.dim_embed)
    gae_model = gae_model.to(device)
    # SN_parameters = in_dim, args.hidden1, args.hidden2
    return train_SN, gae_model, feats, val_edges, val_edges_false, test_edges, test_edges_false

    # create training component
    optimizer = torch.optim.Adam(gae_model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in gae_model.parameters()]))

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full SN
        gae_model.train()

        logits, features = gae_model.forward(SN, feats)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        # loss = F.binary_cross_entropy(logits.view(-1), adj.view(-1))
        # kl_divergence = 0.5 / logits.size(0) * (1 + 2 * gae_model.log_std - gae_model.mean ** 2 - torch.exp(gae_model.log_std) ** 2).sum(1).mean()
        # loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = train_GAE.get_acc(logits, adj)

        val_roc, val_ap = train_GAE.get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = train_GAE.get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print("End of training!", "test_auc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))


if __name__ == '__main__':
    # KG_data_prepare()
    # exit()
    # KG, KG_parameters, triplet, val_edges, val_edges_false, test_edges, test_edges_false = KG_data_prepare()
    KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, test_edges_false_KG = KG_data_prepare()
    # SN, SN_parameters, feats = SN_data_prepare()
    SN, model_SN, feats, val_edges_SN, val_edges_false_SN, test_edges_SN, test_edges_false_SN = SN_data_prepare()
    adj = SN.adjacency_matrix().to_dense().to(device)
    weight_tensor, norm = train_GAE.compute_loss_para(adj, device)

    model = Model_HGT_GAE_GMI(model_KG, model_SN, args.dim_init, args.dim_embed).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=1000, max_lr=1e-3, pct_start=0.05)

    for epoch in range(args.epochs):
        t = time.time()
        # 这里是在整张图上做训练
        model.train()
        negative_graph = data_process_KG.construct_negative_graph(KG, 1, triplet, device)

        pos_score, neg_score, res_mi_KG, res_local_KG, logits, features, res_mi_SN, res_local_SN = model(KG, negative_graph, triplet, SN, feats, adj, args.neg_num, device)
        res_mi_KG_pos, res_mi_KG_neg = res_mi_KG
        res_local_KG_pos, res_local_KG_neg = res_local_KG
        res_mi_SN_pos, res_mi_SN_neg = res_mi_SN
        res_local_SN_pos, res_local_SN_neg = res_local_SN

        loss_KG = data_process_KG.compute_loss(pos_score, neg_score)
        loss_KG_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_KG_pos, res_mi_KG_neg) + args.beta * utils.process.mi_loss_jsd(res_local_KG_pos, res_local_KG_neg)
        # 这里返回的logits已经经过sigmoid，GAE使用整张图作为训练样本
        loss_SN = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        loss_SN_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_SN_pos, res_mi_SN_neg) + args.beta * utils.process.mi_loss_jsd(res_local_SN_pos, res_local_SN_neg)

        # 随机选择paper_id
        loss_align = 0
        nodes = KG.nodes(ntype='paper')
        # 这里的100是从锚节点中选取的训练样本数
        node_align = np.random.choice(nodes.cpu(), 100, replace=False)
        node_align = torch.from_numpy(node_align).to(device)
        # print(KG.nodes['paper'].data['h'].shape)
        # print(features.shape)
        # exit()
        # 向量间的距离作为损失函数
        metrix = KG.nodes['paper'].data['h'][node_align]-features[node_align]
        # 这里对齐没有加翻译层
        loss_align = (metrix*metrix).sum()
        # for id in node_align:
        #     # print(KG.nodes['paper'].data['h'][id])
        #     # print(feats)
        #     vector = KG.nodes['paper'].data['h'][id]-feats[id]
        #     # 返回向量的二阶范数
        #     loss_align += (vector*vector).sum()#np.linalg(vector)



        loss = loss_KG + 0.5*loss_SN + 0.8*loss_align + 0.7*loss_KG_MI + 0.5*loss_SN_MI

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc_KG = data_process_KG.get_acc(pos_score, neg_score)
        train_acc_SN = train_GAE.get_acc(logits, adj)
        val_roc_KG, val_ap_KG = data_process_KG.get_score(KG, val_edges_KG, val_edges_false_KG, triplet)
        val_roc_SN, val_ap_SN = train_GAE.get_scores(val_edges_SN, val_edges_false_SN, logits)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc_KG=",
              "{:.5f}".format(train_acc_KG), "val_roc_KG=", "{:.5f}".format(val_roc_KG), "val_ap_KG=",
              "{:.5f}".format(val_ap_KG),
              "train_acc_SN=", "{:.5f}".format(train_acc_SN), "val_roc_SN=", "{:.5f}".format(val_roc_SN),
              "val_ap_SN=", "{:.5f}".format(val_ap_SN)
              )
        # if (epoch+1) % 10 == 0:
        #     val_roc_KG, val_ap_KG = data_process_KG.get_score(KG, val_edges, val_edges_false, triplet)
        #     val_roc_SN, val_ap_SN = train_GAE.get_scores(val_edges, val_edges_false, logits)
        #     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc_KG=",
        #           "{:.5f}".format(train_acc_KG), "val_roc_KG=", "{:.5f}".format(val_roc_KG), "val_ap_KG=", "{:.5f}".format(val_ap_KG),
        #           "train_acc_SN=", "{:.5f}".format(train_acc_SN), "val_roc_SN=", "{:.5f}".format(val_roc_SN),
        #           "val_ap_SN=", "{:.5f}".format(val_ap_SN)
        #           )
        #     continue
        #     # print('Valid Accuracy:', get_score())
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc_KG=",
        #           "{:.5f}".format(train_acc_KG), "train_acc_SN=", "{:.5f}".format(train_acc_SN))

    test_roc_KG, test_ap_KG = data_process_KG.get_score(KG, test_edges_KG, test_edges_false_KG, triplet)
    test_roc_SN, test_ap_SN = train_GAE.get_scores(test_edges_SN, test_edges_false_SN, logits)
    print("test_roc_KG=", "{:.5f}".format(test_roc_KG), "test_ap_KG=", "{:.5f}".format(test_ap_KG),
          "test_roc_SN=", "{:.5f}".format(test_roc_SN), "test_ap_SN=", "{:.5f}".format(test_ap_SN))