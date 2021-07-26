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
parser.add_argument('--dim_init', '-h_in', type=int, default=400, help='Dim of initial embedding vector.')
parser.add_argument('--dim_embed', '-h_out', type=int, default=16, help='Dim of final embedding vector.')
parser.add_argument('--neg_num', type=int, default=2, help='Number of negtive sampling of each node.')
parser.add_argument('--align_num', type=int, default=100, help='Number of sampling of aligned node.')
parser.add_argument('--align_dist', type=str, default='L2', help='The type of align nodes distance.',
                    choices=['L1', 'L2', 'cos'])
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--cuda', type=int, default=3, help='GPU id to use.')

# 这部分是TGN的参数设置
parser.add_argument("--batch_size", type=int, default=200,
                    help="Size of each batch")
parser.add_argument("--memory_dim", type=int, default=100,
                    help="dimension of memory")
parser.add_argument("--temporal_dim", type=int, default=100,
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
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.85

    # set random Seed
    np.random.seed(2021)
    torch.manual_seed(2021)

    assert not (
        args.fast_mode and args.simple_mode), "you can only choose one sampling mode"
    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    SN = TemporalDataset()
    print(SN)

    # Pre-process data, mask new node in test set from original graph
    num_nodes = SN.num_nodes()
    num_edges = SN.num_edges()

    trainval_div = int(VALID_SPLIT*num_edges)

    # Select new node from test set and remove them from entire graph
    test_split_ts = SN.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([SN.edges()[0][trainval_div:], SN.edges()[
                           1][trainval_div:]]).unique().numpy()
    test_new_nodes = np.random.choice(
        test_nodes, int(0.1*len(test_nodes)), replace=False)

    in_subg = dgl.in_subgraph(SN, test_new_nodes)
    out_subg = dgl.out_subgraph(SN, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    new_node_eid_delete = torch.cat(
        [new_node_in_eid_delete, new_node_out_eid_delete]).unique()

    graph_new_node = copy.deepcopy(SN)
    # relative order preseved
    graph_new_node.remove_edges(new_node_eid_delete)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(SN)
    graph_no_new_node.remove_edges(eid_delete)

    # graph_no_new_node and graph_new_node should have same set of nid

    # Sampler Initialization
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(SN, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(SN, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples)
    # Set Train, validation, test and new node test id
    train_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()))
    valid_seed = torch.arange(int(
        TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    test_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    test_new_node_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges())

    g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        graph_no_new_node, copy_edata=True)
    new_node_g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        graph_new_node, copy_edata=True)
    if not args.fast_mode:
        new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
        g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    train_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              train_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    valid_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              valid_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    test_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                             test_seed,
                                             sampler,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling)

    test_new_node_dataloader = TemporalEdgeDataLoader(graph_new_node,
                                                      test_new_node_seed,
                                                      new_node_sampler if args.fast_mode else sampler,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=new_node_g_sampling)

    edge_dim = SN.edata['feats'].shape[1]
    num_node = SN.num_nodes()

    model_SN = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.dim_embed,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop)

    return model_SN, sampler, train_dataloader, valid_dataloader, test_dataloader, test_new_node_dataloader, new_node_sampler
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
    KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, test_edges_false_KG = KG_data_prepare()
    # SN, SN_parameters, feats = SN_data_prepare()
    model_SN, sampler, train_dataloader, valid_dataloader, test_dataloader, test_new_node_dataloader, new_node_sampler = SN_data_prepare()
    # 损失函数自带sigmoid，说明SN模型没有经过sigmoid层
    criterion_SN = torch.nn.BCEWithLogitsLoss()

    model = Model_HGT_TGN_GMI(model_KG, model_SN, args.dim_init, args.dim_embed).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=1e-3, pct_start=0.05)

    if args.fast_mode:
        sampler.reset()
    for epoch in range(args.epochs):
        t = time.time()
        # 这里是在整张图上做训练
        model.train()
        negative_graph = data_process_KG.construct_negative_graph(KG, 1, triplet, device).to(device)

        pos_score, neg_score, res_mi_KG, res_local_KG = model(KG, negative_graph, triplet, args.neg_num, device)
        res_mi_KG_pos, res_mi_KG_neg = res_mi_KG
        res_local_KG_pos, res_local_KG_neg = res_local_KG
        # res_mi_SN_pos, res_mi_SN_neg = res_mi_SN
        # res_local_SN_pos, res_local_SN_neg = res_local_SN

        loss_KG = data_process_KG.compute_loss(pos_score, neg_score)
        loss_KG_MI = args.alpha * utils.process.mi_loss_jsd(res_mi_KG_pos, res_mi_KG_neg) + args.beta * utils.process.mi_loss_jsd(res_local_KG_pos, res_local_KG_neg)

        # 获取TGN的损失函数，这里不使用batch训练，直接使用总损失函数做训练
        batch_cnt_SN = 0
        # pred_pos, pred_neg = torch.tensor()
        for _, positive_pair_g, negative_pair_g, blocks in train_dataloader:
            # print(blocks)
            # positive_pair_g = torch.tensor(positive_pair_g).to(device)
            # positive_pair_tmp = positive_pair_g#.to(device)
            # negative_pair_tmp = negative_pair_g#.to(device)
            # model_SN = model_SN.to('cpu')
            pred_pos_SN_tmp, pred_neg_SN_tmp = model_SN.embed(
                positive_pair_g, negative_pair_g, blocks, device)
            if batch_cnt_SN == 0:
                pred_pos_SN, pred_neg_SN = pred_pos_SN_tmp, pred_neg_SN_tmp
            else:
                pred_pos_SN = torch.cat((pred_pos_SN, pred_pos_SN_tmp), 0)
                pred_neg_SN = torch.cat((pred_neg_SN, pred_neg_SN_tmp), 0)
            # loss_SN_tmp = criterion_SN(pred_pos, torch.ones_like(pred_pos)) + criterion_SN(pred_neg, torch.zeros_like(pred_neg))
            # loss_SN += float(loss_SN_tmp) * args.batch_size
            # https://blog.csdn.net/qq_24502469/article/details/104559250
            # retain_graph = True if batch_cnt_SN == 0 and not args.fast_mode else False
            # loss.backward(retain_graph=retain_graph)
            # optimizer.step()
            # model_SN.detach_memory()
            # if not args.not_use_memory:
            #     model_SN.update_memory(positive_pair_g, device)
            # if args.fast_mode:
            #     这里会修改self.last_update的位置，改到GPU上
                # sampler.attach_last_update(model_SN.memory.last_update_t)
            # print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
            batch_cnt_SN += 1
        # model_SN = model_SN.to(device)
        loss_SN = criterion_SN(pred_pos_SN, torch.ones_like(pred_pos_SN)) + criterion_SN(pred_neg_SN, torch.zeros_like(pred_neg_SN))
        # loss_SN_MI = args.alpha * utils.process.mi_loWss_jsd(res_mi_SN_pos, res_mi_SN_neg) + args.beta * utils.process.mi_loss_jsd(res_local_SN_pos, res_local_SN_neg)

        # # 锚节点对齐，随机选择paper_id
        # # loss_align = 0
        # nodes = KG.nodes(ntype='paper')
        # node_align = np.random.choice(nodes.cpu(), args.align_num, replace=False)
        # node_align = torch.from_numpy(node_align).to(device)
        # # print(KG.nodes['paper'].data['h'].shape)
        # # print(features.shape)
        # # exit()
        # if args.align_dist == 'L2':
        #     # 向量间的距离作为损失函数
        #     metrix = KG.nodes['paper'].data['h'][node_align]-trans_SN[node_align]
        #     # 这里对齐没有加翻译层
        #     loss_align = (metrix*metrix).sum()
        # elif args.align_dist == 'L1':
        #     metrix = KG.nodes['paper'].data['h'][node_align]-trans_SN[node_align]
        #     loss_align = torch.abs(metrix).sum()
        # elif args.align_dist == 'cos':
        #     # dim=1，计算行向量的相似度
        #     res = torch.cosine_similarity(KG.nodes['paper'].data['h'][node_align], trans_SN[node_align], dim=1)
        #     # 余弦相似度在[-1, 1]间，为1相似度高，损失函数就小
        #     loss_align = -res.sum()/args.align_num

        loss = loss_KG + loss_SN# + 0.8*loss_align + 0.7*loss_KG_MI + 0.5*loss_SN_MI

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc_KG = data_process_KG.get_acc(pos_score, neg_score)
        val_auc_KG, val_ap_KG = data_process_KG.get_score(KG, val_edges_KG, val_edges_false_KG, triplet)
        # 社交网络的指标测试
        train_acc_SN = data_process_KG.get_acc(pred_pos_SN, pred_neg_SN)
        val_ap_SN, val_auc_SN = train_TGN.test_val(
            model_SN, valid_dataloader, sampler, criterion_SN, args, device)
        memory_checkpoint = model_SN.store_memory()
        if args.fast_mode:
            new_node_sampler.sync(sampler)
        test_ap_SN, test_auc_SN = train_TGN.test_val(
            model_SN, test_dataloader, sampler, criterion_SN, args, device)
        model_SN.restore_memory(memory_checkpoint)
        if args.fast_mode:
            sample_nn = new_node_sampler
        else:
            sample_nn = sampler
        nn_test_ap_SN, nn_test_auc_SN = train_TGN.test_val(
            model_SN, test_new_node_dataloader, sample_nn, criterion_SN, args, device)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc_KG=",
              "{:.5f}".format(train_acc_KG), "val_roc_KG=", "{:.5f}".format(val_auc_KG), "val_ap_KG=",
              "{:.5f}".format(val_ap_KG),
              "train_acc_SN=", "{:.5f}".format(train_acc_SN), "val_roc_SN=", "{:.5f}".format(val_auc_SN),
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

    test_auc_KG, test_ap_KG = data_process_KG.get_score(KG, test_edges_KG, test_edges_false_KG, triplet)
    # test_roc_SN, test_ap_SN = train_GAE.get_scores(test_edges_SN, test_edges_false_SN, logits)
    print("test_roc_KG=", "{:.5f}".format(test_auc_KG), "test_ap_KG=", "{:.5f}".format(test_ap_KG),
          "test_roc_SN=", "{:.5f}".format(test_auc_SN), "test_ap_SN=", "{:.5f}".format(test_ap_SN))