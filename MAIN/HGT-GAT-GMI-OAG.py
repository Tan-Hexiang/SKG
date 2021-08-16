# 作为后续模型代码的参考，三个任务都划分了训练集、验证集和测试集，且使用了GMI，训练时增加了负采样和early stopping
import sys
sys.path.append('../HGT-DGL')
sys.path.append('../GAE')
sys.path.append('../GMI')
# from train_paper_venue import *
import data_process
# from utils import EarlyStopMonitor
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
parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--dim_init', '-h_in', type=int, default=768, help='Dim of initial embedding vector.')
parser.add_argument('--dim_embed', '-h_out', type=int, default=16, help='Dim of final embedding vector.')
parser.add_argument('--neg_num', type=int, default=2, help='Number of negtive sampling of each node.')
parser.add_argument('--margin', type=int, default=1, help='The margin of alignment for entities.')
# parser.add_argument('--align_num', type=int, default=100, help='Number of sampling of aligned node.')
# 用cos效果好的比较明显，但是实体对齐没有啥变化
parser.add_argument('--align_dist', type=str, default='cos', help='The type of align nodes distance.',
                    choices=['L1', 'L2', 'cos'])
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--cuda', type=int, default=3, help='GPU id to use.')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio.')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='Valid set ratio.')
parser.add_argument('--tolerance', type=float, default=1e-3,  help='toleratd margainal improvement for early stopper')
args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
# device = torch.device('cpu')

def KG_data_prepare():
    KG, KG_forward, KG_backward = data_process.OAG_KG_ReadData()
    print(KG)
    # print(KG.adjacency_matrix(etype='written-by'))
    # exit()
    triplet = ('author', 'study', 'field')
    # 这里的邻接矩阵作用只是提供idx的范围，行数是源节点，列数是目标节点
    # KG = KG.to(device)
    adj_orig = KG.adjacency_matrix(etype=triplet[1]).to_dense()
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = data_process_KG.mask_test_edges_dgl(KG, adj_orig, triplet[1], args.train_ratio, args.valid_ratio)
    train_edge_idx = torch.tensor(train_edge_idx)#.to(device)

    # 老版本preserve_nodes=True会报错
    train_graph = KG.edge_subgraph({
        # 把author/paper这类和各种节点都有连边的点放在源节点的位置，猜测是multi_update_all函数只更新尾实体，所以如果尾实体是单独的节点类型，
        # 会没有添加't'属性
        ('author', 'study', 'field'): train_edge_idx,
        ('author', 'in', 'affiliation'): list(range(KG.number_of_edges('in'))),
        ('affiliation', 'has', 'author'): list(range(KG.number_of_edges('has'))),
        ('author', 'contribute', 'venue'): list(range(KG.number_of_edges('contribute'))),
        ('venue', 'be-contributed', 'author'): list(range(KG.number_of_edges('be-contributed'))),
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
    # node_features = {}
    # for ntype in KG.ntypes:
    #     emb = nn.Parameter(torch.Tensor(KG.number_of_nodes(ntype), args.dim_init), requires_grad=False)  # .to(device)
    #     nn.init.xavier_uniform_(emb)
    #     KG.nodes[ntype].data['feature'] = emb
    #     node_features[ntype] = emb

    KG = KG.to(device)
    model_KG = HGT_PF(KG, args.dim_init, args.hidden1, args.dim_embed, n_layers=2, n_heads=4, use_norm=True).to(device)
    # KG_parameters = KG, 400, 200, 16, 2, 4, True
    return KG, model_KG, triplet, val_edges, val_edges_false, test_edges, test_edges_false, KG_forward, KG_backward

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
    SN, SN_forward, SN_backward = data_process.OAG_SN_ReadData()
    print(SN)
    # 生成每个节点的特征向量
    # emb = nn.Parameter(torch.Tensor(len(SN.nodes()), args.dim_init), requires_grad=False)#.to(device)
    # nn.init.xavier_uniform_(emb)
    # SN.ndata['feature'] = emb
    feats = SN.ndata.pop('feature').to(device)
    in_dim = feats.shape[-1]

    # generate input
    adj_orig = SN.adjacency_matrix().to_dense()

    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = preprocess.mask_test_edges_dgl(SN, adj_orig, args.train_ratio, args.valid_ratio)

    SN = SN.to(device)

    # create train SN
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_SN = dgl.edge_subgraph(SN, train_edge_idx, preserve_nodes=True).to(device)
    train_SN = train_SN.to(device)
    # adj = train_SN.adjacency_matrix().to_dense().to(device)

    # compute loss parameters
    # weight_tensor, norm = train_GAE.compute_loss_para(adj, device)

    # create model
    gae_model = model_GAE.GAEModel(in_dim, args.hidden1, args.dim_embed)
    gae_model = gae_model.to(device)
    # SN_parameters = in_dim, args.hidden1, args.hidden2
    return train_SN, gae_model, feats, val_edges, val_edges_false, test_edges, test_edges_false, SN_forward, SN_backward

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
    KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, test_edges_false_KG, KG_forward, KG_backward = KG_data_prepare()
    # SN, SN_parameters, feats = SN_data_prepare()
    SN, model_SN, feats, val_edges_SN, val_edges_false_SN, test_edges_SN, test_edges_false_SN, SN_forward, SN_backward = SN_data_prepare()

    # 生成实体对齐的训练集
    nodes = set(KG_forward.values()) & set(SN_forward.values())
    nodes_tmp = [KG_backward[node] for node in nodes]
    # 这里nodes保存的是KG中的节点id
    nodes = np.array(nodes_tmp)
    np.random.shuffle(nodes)
    # OAG中实体对齐的id需要进行转化，因为两张图的id排列是不一样的
    # 因为有些KG中的节点不一定在SN中，有可能是因为引用数高，但是合作少，所以没有和SN中的节点产生链接
    # 从锚节点中选取训练样本数
    # node_align_KG_train = np.random.choice(nodes, int(len(nodes)*args.train_ratio), replace=False)
    train_idx = int(len(nodes)*args.train_ratio)
    valid_idx = int(len(nodes)*(args.train_ratio+args.valid_ratio))
    node_align_KG_train = nodes[:train_idx]
    node_align_KG_valid = nodes[train_idx:valid_idx]
    node_align_KG_test  = nodes[valid_idx:]
    node_align_SN_train = [SN_backward[KG_forward[node]] for node in node_align_KG_train]
    node_align_SN_valid = [SN_backward[KG_forward[node]] for node in node_align_KG_valid]
    node_align_SN_test  = [SN_backward[KG_forward[node]] for node in node_align_KG_test]
    # CPU->GPU
    node_align_KG_train = torch.from_numpy(node_align_KG_train).to(device)
    node_align_KG_valid = torch.from_numpy(node_align_KG_valid).to(device)
    node_align_KG_test  = torch.from_numpy(node_align_KG_test).to(device)
    node_align_SN_train = torch.tensor(node_align_SN_train).to(device)
    node_align_SN_valid = torch.tensor(node_align_SN_valid).to(device)
    node_align_SN_test  = torch.tensor(node_align_SN_test).to(device)
    # repeat_interleave(args.neg_num)生成的样本是相同节点相邻，即279, 279, 497, 497, 287, 287
    # repeat生成的是两个相同的数组拼接
    node_align_KG_train = node_align_KG_train.repeat(args.neg_num)
    node_align_SN_train = node_align_SN_train.repeat(args.neg_num)
    # print(torch.cat((node_align_KG_valid, node_align_KG_test), 0))

    adj_SN = SN.adjacency_matrix().to_dense().to(device)
    weight_tensor_SN, norm_SN = train_GAE.compute_loss_para(adj_SN, device)

    model = Model_HGT_GAE_GMI(model_KG, model_SN, args.dim_init, args.dim_embed).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=1e-3, pct_start=0.05)
    early_stopper = data_process.EarlyStopMonitor(tolerance=args.tolerance)

    for epoch in range(args.epochs):
        t = time.time()
        # 这里是在整张图上做训练
        model.train()
        negative_graph = data_process_KG.construct_negative_graph(KG, 1, triplet, device)

        pos_score, neg_score, res_mi_KG, res_local_KG, logits, features, trans_SN, res_mi_SN, res_local_SN = \
            model(KG, negative_graph, triplet, SN, feats, adj_SN, args.neg_num, device)
        res_mi_KG_pos, res_mi_KG_neg = res_mi_KG
        res_local_KG_pos, res_local_KG_neg = res_local_KG
        res_mi_SN_pos, res_mi_SN_neg = res_mi_SN
        res_local_SN_pos, res_local_SN_neg = res_local_SN

        loss_KG = data_process_KG.compute_loss(pos_score, neg_score)
        loss_KG_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_KG_pos, res_mi_KG_neg) + args.beta * utils.process.mi_loss_jsd(res_local_KG_pos, res_local_KG_neg)
        # 这里返回的logits已经经过sigmoid，GAE使用整张图作为训练样本
        loss_SN = norm_SN * F.binary_cross_entropy(logits.view(-1), adj_SN.view(-1), weight=weight_tensor_SN)
        loss_SN_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_SN_pos, res_mi_SN_neg) + args.beta * utils.process.mi_loss_jsd(res_local_SN_pos, res_local_SN_neg)

        # 随机选择author_id
        # loss_align = 0
        # # 这里添加了翻译层
        # 损失函数增加负采样，这里的负样本从社交网络中选取，因为社交网络的作者节点多
        node_align_KG_neg = node_align_KG_train#.repeat_interleave(args.neg_num)
        node_align_SN_neg = torch.randint(0, SN.num_nodes(), (len(node_align_SN_train),)).to(device)

        loss_align = 0
        if args.align_dist == 'L2':
            # 向量间的距离作为损失函数
            metrix_pos = KG.nodes['author'].data['h'][node_align_KG_train]-trans_SN[node_align_SN_train]
            metrix_neg = KG.nodes['author'].data['h'][node_align_KG_neg]-trans_SN[node_align_SN_neg]
            res_pos = torch.sqrt(torch.sum(metrix_pos*metrix_pos, dim=1))
            res_neg = torch.sqrt(torch.sum(metrix_neg*metrix_neg, dim=1))
            results = args.margin+res_pos-res_neg
            for result in results:
                loss_align += max(0, result)
            # loss_align = (metrix*metrix).sum()/len(node_align_KG_train)
        elif args.align_dist == 'L1':
            # metrix = KG.nodes['author'].data['h'][node_align_KG_train]-trans_SN[node_align_SN_train]
            # loss_align = torch.abs(metrix).sum()/len(node_align_KG_train)
            metrix_pos = KG.nodes['author'].data['h'][node_align_KG_train]-trans_SN[node_align_SN_train]
            metrix_neg = KG.nodes['author'].data['h'][node_align_KG_neg]-trans_SN[node_align_SN_neg]
            res_pos = torch.sum(torch.abs(metrix_pos), dim=1)
            res_neg = torch.sum(torch.abs(metrix_neg), dim=1)
            results = args.margin+res_pos-res_neg
            for result in results:
                loss_align += max(0, result)
        elif args.align_dist == 'cos':
            # dim=1，计算行向量的相似度
            res_pos = torch.cosine_similarity(KG.nodes['author'].data['h'][node_align_KG_train], trans_SN[node_align_SN_train], dim=1)
            res_neg = torch.cosine_similarity(KG.nodes['author'].data['h'][node_align_KG_neg], trans_SN[node_align_SN_neg], dim=1)
            # 余弦相似度在[-1, 1]间，为1相似度高，损失函数就小
            results = args.margin-res_pos+res_neg
            for result in results:
                loss_align += max(0, result)
            # loss_align = -res.sum()/args.align_num
        loss_align /= len(node_align_KG_train)

        loss = loss_KG + 0.5*loss_SN + 0.7*loss_KG_MI + 0.5*loss_SN_MI + 0.8*loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # 知识图谱链接预测的结果
            train_acc_KG = data_process_KG.get_acc(pos_score, neg_score)
            val_roc_KG, val_ap_KG = data_process_KG.get_score(KG, val_edges_KG, val_edges_false_KG, triplet)
            # 社交网络链接预测的结果
            train_acc_SN = train_GAE.get_acc(logits, adj_SN)
            val_roc_SN, val_ap_SN = train_GAE.get_scores(val_edges_SN, val_edges_false_SN, logits)
            # 实体对齐的指标MRR，hits@10
            # 知识图谱的节点向量KG.nodes['author'].data['h'][node_align_KG_train]
            # 社交网络的节点向量trans_SN[node_align_SN_train]
            val_MRR_align, val_hits5_align = data_process.align_scores(KG.nodes['author'].data['h'], trans_SN,node_align_KG_valid,
                                                                 node_align_SN_valid, 5 , args.align_dist)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_ACC_KG=",
              "{:.5f}".format(train_acc_KG), "val_AUC_KG=", "{:.5f}".format(val_roc_KG), "val_AP_KG=",
              "{:.5f}".format(val_ap_KG),
              "train_ACC_SN=", "{:.5f}".format(train_acc_SN), "val_AUC_SN=", "{:.5f}".format(val_roc_SN),
              "val_AP_SN=", "{:.5f}".format(val_ap_SN), 'val_MRR_align=', '{:.5f}'.format(val_MRR_align),
              "val_hits@5_align=", '{:.5f}'.format(val_hits5_align)
              )
        # early stopping
        if early_stopper.early_stop_check(val_ap_KG, model):
            print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            print(f'Loading the best model at epoch {early_stopper.best_epoch}')
            model = early_stopper.model
            break

    model(KG, negative_graph, triplet, SN, feats, adj_SN, args.neg_num, device)
    test_roc_KG, test_ap_KG = data_process_KG.get_score(KG, test_edges_KG, test_edges_false_KG, triplet)
    test_roc_SN, test_ap_SN = train_GAE.get_scores(test_edges_SN, test_edges_false_SN, logits)
    test_MRR_align, test_hits5_align = data_process.align_scores(KG.nodes['author'].data['h'], trans_SN,
                                                               node_align_KG_valid,
                                                               node_align_SN_valid, 5, args.align_dist)
    print("test_roc_KG=", "{:.5f}".format(test_roc_KG), "test_ap_KG=", "{:.5f}".format(test_ap_KG),
          "test_roc_SN=", "{:.5f}".format(test_roc_SN), "test_ap_SN=", "{:.5f}".format(test_ap_SN),
          'test_MRR_align=', '{:.5f}'.format(test_MRR_align), "test_hits@5_align=", '{:.5f}'.format(test_hits5_align))