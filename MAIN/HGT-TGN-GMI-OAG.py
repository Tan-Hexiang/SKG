# 用HGT做个实验，两个HGT都对异质图做操作，一个异质图只有论文和作者，一个异质图只有论文、会议和领域
import sys
sys.path.append('../HGT-DGL')
sys.path.append('../TGN')
sys.path.append('../GMI')

from model import *
import torch
import numpy as np
from warnings import filterwarnings
import time
filterwarnings("ignore")
import data_process

import data_process_KG
from model_HGT import *

import traceback
import copy
import data_preprocess_TGN
import train_TGN
from tgn import TGN
from data_preprocess_TGN import TemporalWikipediaDataset, TemporalRedditDataset, TemporalDataset
from dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                         SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.metrics import average_precision_score, roc_auc_score


import argparse
parser = argparse.ArgumentParser(description='HGT-TGN-GMI-OAG')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--dim_init', '-h_in', type=int, default=768, help='Dim of initial embedding vector.')
parser.add_argument('--dim_embed', '-h_out', type=int, default=16, help='Dim of final embedding vector.')
parser.add_argument('--neg_num', type=int, default=2, help='Number of negtive sampling of each node.')
parser.add_argument('--align_num', type=int, default=100, help='Number of sampling of aligned node.')
parser.add_argument('--align_dist', type=str, default='L2', help='The type of align nodes distance.',
                    choices=['L1', 'L2', 'cos'])
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--cuda', type=int, default=0, help='GPU id to use.')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Training set ratio.')

# 这部分是TGN的参数设置
# TGN每次子图的边数
parser.add_argument("--batch_size", type=int, default=200,
                    help="Size of each batch")
parser.add_argument("--memory_dim", type=int, default=32,
                    help="dimension of memory")
parser.add_argument("--temporal_dim", type=int, default=32,
                    help="Temporal dimension for time encoding")
parser.add_argument("--memory_updater", type=str, default='gru',
                    help="Recurrent unit for memory update")
parser.add_argument("--aggregator", type=str, default='last',
                    help="Aggregation method for memory update")
parser.add_argument("--n_neighbors", type=int, default=10,
                    help="number of neighbors while doing embedding")
parser.add_argument("--sampling_method", type=str, default='topk',
                    help="In embedding how node aggregate from its neighor")
parser.add_argument("--num_heads", type=int, default=8,
                    help="Number of heads for multihead attention mechanism")
parser.add_argument("--fast_mode", action="store_true", default=True,
                    help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
parser.add_argument("--simple_mode", action="store_true", default=False,
                    help="Simple Mode directly delete the temporal edges from the original static graph")
parser.add_argument("--num_negative_samples", type=int, default=1,
                    help="number of negative samplers per positive samples")
parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
parser.add_argument("--not_use_memory", action="store_true", default=False,
                    help="Enable memory for TGN Model disable memory for TGN Model")


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
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = data_process_KG.mask_test_edges_dgl(KG, adj_orig, triplet[1])
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
    # print(train_graph)
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
    # 这里只考虑transductive，不考虑new_node
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.85

    # set random Seed
    np.random.seed(2021)
    torch.manual_seed(2021)

    assert not (
        args.fast_mode and args.simple_mode), "you can only choose one sampling mode"
    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    SN, SN_forward, SN_backward = data_process.OAG_SN_ReadData()
    print(SN)

    # Pre-process data, mask new node in test set from original graph
    num_nodes = SN.num_nodes()
    num_edges = SN.num_edges()

    train_div = int(TRAIN_SPLIT*num_edges)
    trainval_div = int(VALID_SPLIT*num_edges)

    # Sampler Initialization
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(SN, fan_out)
        # new_node_sampler = SimpleTemporalSampler(SN, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(SN, k=args.n_neighbors)
        # new_node_sampler = FastTemporalSampler(SN, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples)
    # Set Train, validation, test and new node test id
    train_seed = torch.arange(train_div)
    valid_seed = torch.arange(train_div, trainval_div)
    test_seed = torch.arange(trainval_div, num_edges)

    g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        SN, copy_edata=True)
    new_node_g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        SN, copy_edata=True)
    if not args.fast_mode:
        new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
        g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    # 这里设置每次把整张图都采出来
    # 这个train_dataloader在第二轮的时候会生成一个节点更大的图，节点数都超过了SN的节点数
    train_dataloader = TemporalEdgeDataLoader(SN,
                                              train_seed,
                                              sampler,
                                              batch_size=len(train_seed),
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    valid_dataloader = TemporalEdgeDataLoader(SN,
                                              valid_seed,
                                              sampler,
                                              batch_size=len(valid_seed),
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    test_dataloader = TemporalEdgeDataLoader(SN,
                                             test_seed,
                                             sampler,
                                             batch_size=len(test_seed),
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling)

    edge_dim = SN.edata['feature'].shape[1]
    num_node = SN.num_nodes()

    model_SN = TGN(node_features=SN.ndata['feature'],
                edge_feat_dim=edge_dim,
                memory_dim=args.dim_init,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.dim_embed,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop)

    return SN.to(device), model_SN, sampler, train_dataloader, valid_dataloader, test_dataloader, SN_forward, SN_backward
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Implement Logging mechanism
    f = open("logging.txt", 'w')
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_loss = train_TGN.train(model, train_dataloader, sampler,
                               criterion, optimizer, args)

            val_ap, val_auc = train_TGN.test_val(
                model, valid_dataloader, sampler, criterion, args)
            memory_checkpoint = model.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            test_ap, test_auc = train_TGN.test_val(
                model, test_dataloader, sampler, criterion, args)
            model.restore_memory(memory_checkpoint)
            if args.fast_mode:
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap, nn_test_auc = train_TGN.test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                i, train_loss, val_ap, val_auc))
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                i, nn_test_ap, nn_test_auc))

            f.writelines(log_content)
            model.reset_memory()
            if i < args.epochs-1 and args.fast_mode:
                sampler.reset()
            print(log_content[0], log_content[1], log_content[2])
    except:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")

if __name__ == '__main__':
    # SN_data_prepare()
    # exit()
    # KG, KG_parameters, triplet, val_edges, val_edges_false, test_edges, test_edges_false = KG_data_prepare()
    KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, test_edges_false_KG, KG_forward, KG_backward = KG_data_prepare()
    # SN, SN_parameters, feats = SN_data_prepare()
    SN, model_SN, sampler, train_dataloader, valid_dataloader, test_dataloader, SN_forward, SN_backward = SN_data_prepare()
    # SN = SN.to(device)

    # 生成正样本图和block，不然TrainDataLoader会生成更大的图，不知道为什么
    # for _, positive_pair_g, _, blocks in train_dataloader:
    #     pass

    # 生成实体对齐的训练集
    nodes = set(KG_forward.values()) & set(SN_forward.values())
    nodes_tmp = [KG_backward[node] for node in nodes]
    nodes = np.array(nodes_tmp)
    # OAG中实体对齐的id需要进行转化，因为两张图的id排列是不一样的
    # 因为有些KG中的节点不一定在SN中，有可能是因为引用数高，但是合作少，所以没有和SN中的节点产生链接
    # 从锚节点中选取训练样本数
    node_align_KG = np.random.choice(nodes, int(len(nodes)*args.train_ratio), replace=False)
    # for node in node_align_KG:
    #     node_align_SN.append(SN_backward[KG_forward[node]])
    node_align_SN = [SN_backward[KG_forward[node]] for node in node_align_KG]
    node_align_KG = torch.from_numpy(node_align_KG).to(device)
    node_align_SN = torch.tensor(node_align_SN).to(device)

    # 损失函数自带sigmoid，说明SN模型没有经过sigmoid层
    criterion_SN = torch.nn.BCEWithLogitsLoss()

    model = Model_HGT_TGN_GMI(model_KG, model_SN, args.dim_init, args.dim_embed).to(device)
    # print(model_SN.device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=1e-3, pct_start=0.05)

    if args.fast_mode:
        sampler.reset()
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        t = time.time()
        # 这里是在整张图上做训练
        model.train()
        negative_graph = data_process_KG.construct_negative_graph(KG, 1, triplet, device).to(device)

        # 获取TGN的损失函数，这里不使用batch训练，直接使用总损失函数做训练
        batch_cnt_SN = 0
        # pred_pos, pred_neg = torch.tensor()
        # for _, positive_pair_g, negative_pair_g, blocks in train_dataloader:
        #
        #     pred_pos_SN_tmp, pred_neg_SN_tmp, emb_SN, res_mi_SN, res_local_SN = model_SN.embed(
        #         positive_pair_g, negative_pair_g, blocks, device)
        #     if batch_cnt_SN == 0:
        #         pred_pos_SN, pred_neg_SN = pred_pos_SN_tmp, pred_neg_SN_tmp
        #     else:
        #         pred_pos_SN = torch.cat((pred_pos_SN, pred_pos_SN_tmp), 0)
        #         pred_neg_SN = torch.cat((pred_neg_SN, pred_neg_SN_tmp), 0)
        #     batch_cnt_SN += 1
        # 生成TGN的正负样本图
        for _, positive_pair_g, negative_pair_g, blocks in train_dataloader:
            pass
        positive_pair_g = positive_pair_g.to(device)
        negative_pair_g = negative_pair_g.to(device)
        blocks[0] = blocks[0].to(device)
        pos_score, neg_score, res_mi_KG, res_local_KG, pred_pos_SN, pred_neg_SN, trans_SN, res_mi_SN, res_local_SN = model(KG, negative_graph, triplet, args.neg_num, positive_pair_g, negative_pair_g, blocks, device)
        res_mi_KG_pos, res_mi_KG_neg = res_mi_KG
        res_local_KG_pos, res_local_KG_neg = res_local_KG

        loss_KG = data_process_KG.compute_loss(pos_score, neg_score)
        loss_KG_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_KG_pos, res_mi_KG_neg) + args.beta * utils.process.mi_loss_jsd(res_local_KG_pos, res_local_KG_neg)

        # model_SN = model_SN.to(device)
        loss_SN = criterion_SN(pred_pos_SN, torch.ones_like(pred_pos_SN)) + criterion_SN(pred_neg_SN, torch.zeros_like(pred_neg_SN))
        res_mi_SN_pos, res_mi_SN_neg = res_mi_SN
        res_local_SN_pos, res_local_SN_neg = res_local_SN
        loss_SN_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_SN_pos, res_mi_SN_neg) + args.beta * utils.process.mi_loss_jsd(res_local_SN_pos, res_local_SN_neg)

        # 随机选择author_id

        # # 这里没有使用翻译层
        if args.align_dist == 'L2':
            # 向量间的距离作为损失函数
            metrix = KG.nodes['author'].data['h'][node_align_KG]-trans_SN[node_align_SN]
            loss_align = (metrix*metrix).sum()
        elif args.align_dist == 'L1':
            metrix = KG.nodes['author'].data['h'][node_align_KG]-trans_SN[node_align_SN]
            loss_align = torch.abs(metrix).sum()
        elif args.align_dist == 'cos':
            # dim=1，计算行向量的相似度
            res = torch.cosine_similarity(KG.nodes['author'].data['h'][node_align_KG], trans_SN[node_align_SN], dim=1)
            # 余弦相似度在[-1, 1]间，为1相似度高，损失函数就小
            loss_align = -res.sum()/args.align_num

        loss = loss_KG + loss_SN + 0.8*loss_align + 0.7*loss_KG_MI + 0.5*loss_SN_MI

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 删除无用的变量
        del negative_pair_g, blocks,  res_mi_KG, res_local_KG, trans_SN, res_mi_SN, res_local_SN, metrix

        if not args.not_use_memory:
            model_SN.update_memory(positive_pair_g, device)
        if args.fast_mode:
            sampler.attach_last_update(model_SN.memory.last_update_t)

        with torch.no_grad():
            train_acc_KG = data_process_KG.get_acc(pos_score, neg_score)
            val_auc_KG, val_ap_KG = data_process_KG.get_score(KG, val_edges_KG, val_edges_false_KG, triplet)
            # 社交网络的指标测试
            train_acc_SN = data_process_KG.get_acc(pred_pos_SN, pred_neg_SN)
            val_ap_SN, val_auc_SN = train_TGN.test_val(
                model_SN, valid_dataloader, sampler, criterion_SN, args, device)
            memory_checkpoint = model_SN.store_memory()
            test_ap_SN, test_auc_SN = train_TGN.test_val(
                model_SN, test_dataloader, sampler, criterion_SN, args, device)
            model_SN.restore_memory(memory_checkpoint)
        # model_SN.reset_memory(SN.ndata['feature'])
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc_KG=",
              "{:.5f}".format(train_acc_KG), "val_roc_KG=", "{:.5f}".format(val_auc_KG), "val_ap_KG=",
              "{:.5f}".format(val_ap_KG),
              "train_acc_SN=", "{:.5f}".format(train_acc_SN), "val_roc_SN=", "{:.5f}".format(val_auc_SN),
              "val_ap_SN=", "{:.5f}".format(val_ap_SN)
              )

        # 删除无用的变量
        del positive_pair_g,  pos_score, neg_score, pred_pos_SN, pred_neg_SN

    test_auc_KG, test_ap_KG = data_process_KG.get_score(KG, test_edges_KG, test_edges_false_KG, triplet)
    # test_roc_SN, test_ap_SN = train_GAE.get_scores(test_edges_SN, test_edges_false_SN, logits)
    print("test_roc_KG=", "{:.5f}".format(test_auc_KG), "test_ap_KG=", "{:.5f}".format(test_ap_KG),
          "test_roc_SN=", "{:.5f}".format(test_auc_SN), "test_ap_SN=", "{:.5f}".format(test_ap_SN))