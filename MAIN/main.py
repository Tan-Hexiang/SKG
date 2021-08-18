from parser import *
import sys
sys.path.extend(['../RGCN', '../HGT', '../GCN', '../GAT', '../GAE', '../GMI'])

# 导入主实验模块
import data_process_MAIN
from model import *
import torch
import numpy as np
import time
import data_process_KG
import data_process_SN

# 导入KG模块
import model_RGCN
import model_HGT

# 导入SN模块
import model_GAE
import model_GCN
import model_GAT

from warnings import filterwarnings
filterwarnings("ignore")

args, sys_argv = get_args()
print(args)
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

def KG_data_prepare():
    # KG数据读取
    if args.dataset == 'OAG':
        if args.task == 'KG_NC':
            KG, KG_forward, KG_backward = data_process_MAIN.OAG_KG_ReadData_NC(args)
            category = 'author'
        else:
            KG, KG_forward, KG_backward = data_process_MAIN.OAG_KG_ReadData_LP(args)
            triplet = ('author', 'study', 'field')
    elif args.dataset == 'WDT':
        if args.task == 'KG_NC':
            KG, KG_forward, KG_backward = data_process_MAIN.WDT_KG_ReadData_NC(args)
            category = 'person'
        else:
            KG, KG_forward, KG_backward = data_process_MAIN.WDT_KG_ReadData_LP(args)
            triplet = ('person', 'employ', 'affiliation')
    else:
        assert (args.dataset in ['OAG', 'WDT'])

    if args.task == 'KG_NC':
        # 需要分类的节点类型
        train_mask = KG.nodes[category].data.pop('train_mask')
        valid_mask = KG.nodes[category].data.pop('valid_mask')
        test_mask = KG.nodes[category].data.pop('test_mask')
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze()
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
        labels = KG.nodes[category].data.pop('label')
        num_classes = labels.shape[1]

        KG = KG.to(device)
        labels = labels.to(device)
        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)

        # create model
        if args.KG_model == 'RGCN':
            model_KG = model_RGCN.RGCN_NC(KG,
                                       in_dim=args.hidden_dim,# input dim现在必须得和隐藏层维度相同，因为i2h层没有办法把维度降下来
                                       hidden_dim=args.hidden_dim,
                                       out_dim=num_classes,
                                       num_hidden_layers=args.n_layers - 2,
                                       dropout=args.dropout,
                                       use_self_loop=args.use_self_loop)
        elif args.KG_model == 'HGT':
            model_KG = model_HGT.HGT_NC(KG,
                                        in_dim=args.in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=num_classes,
                                        n_layers=2,
                                        n_heads=4,
                                        use_norm=True)
        else:
            assert (args.KG_model in ['RGCN', 'HGT'])
        model_KG = model_KG.to(device)
        print(KG)
        return KG, model_KG, category, labels, train_idx, valid_idx, test_idx, KG_forward, KG_backward
    else:
        if args.task == 'KG_LP':
            # KG_LP时，KG需要采样，其余时候不需要
            # 这里的邻接矩阵作用只是提供idx的范围，行数是源节点，列数是目标节点
            adj_orig = KG.adjacency_matrix(etype=triplet[1]).to_dense()
            train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = data_process_KG.mask_test_edges_LP(
                KG, adj_orig, triplet[1], args.train_ratio, args.valid_ratio)
            train_edge_idx = torch.tensor(train_edge_idx)

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
            KG = train_graph
        else:
            val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

        KG.node_dict = {}
        KG.edge_dict = {}
        # 给每个类型加上id，从0开始
        for ntype in KG.ntypes:
            KG.node_dict[ntype] = len(KG.node_dict)
        for etype in KG.etypes:
            KG.edge_dict[etype] = len(KG.edge_dict)
            # 貌似dgl的图在to(device)后就不能进行更改了
            KG.edges[etype].data['id'] = torch.ones(KG.number_of_edges(etype), dtype=torch.long) * KG.edge_dict[etype]

        KG = KG.to(device)
        if args.KG_model == 'RGCN':
            model_KG = model_RGCN.RGCN_LP(KG,
                                          in_dim=args.hidden_dim,
                                          hidden_dim=args.hidden_dim,
                                          out_dim=args.hidden_dim,
                                          num_hidden_layers=args.n_layers,
                                          dropout=args.dropout,
                                          use_self_loop=args.use_self_loop)
        elif args.KG_model == 'HGT':
            model_KG = model_HGT.HGT_LP(KG,
                                        in_dim=args.in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=args.out_dim,
                                        n_layers=2,
                                        n_heads=4,
                                        use_norm=True)
        else:
            assert (args.KG_model in ['RGCN', 'HGT'])
        model_KG = model_KG.to(device)
        print(KG)
        return KG, model_KG, triplet, val_edges, val_edges_false, test_edges, test_edges_false, KG_forward, KG_backward

def SN_data_prepare():
    # SN数据读取
    if args.dataset == 'OAG':
        if args.task == 'SN_NC':
            SN, SN_forward, SN_backward = data_process_MAIN.OAG_SN_ReadData_NC(args)
        else:
            SN, SN_forward, SN_backward = data_process_MAIN.OAG_SN_ReadData_LP(args)
    elif args.dataset == 'WDT':
        if args.task == 'SN_NC':
            SN, SN_forward, SN_backward = data_process_MAIN.WDT_SN_ReadData_NC(args)
        else:
            SN, SN_forward, SN_backward = data_process_MAIN.WDT_SN_ReadData_LP(args)
    else:
        assert (args.dataset in ['OAG', 'WDT'])

    if args.task == 'SN_NC':
        pass
    else:
        feature_SN = SN.ndata.pop('feature').to(device)
        in_dim = feature_SN.shape[-1]
        if args.task == 'SN_LP':
            # generate input
            adj_orig = SN.adjacency_matrix().to_dense()

            # build test set with 10% positive links
            train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = \
                data_process_SN.mask_test_edges_LP(SN, adj_orig, args.train_ratio, args.valid_ratio)
            SN = SN.to(device)

            # create train SN
            train_edge_idx = torch.tensor(train_edge_idx).to(device)
            train_SN = dgl.edge_subgraph(SN, train_edge_idx, preserve_nodes=True).to(device)
            SN = train_SN
        else:
            val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

        SN = SN.to(device)
        # create model
        if args.SN_model == 'GCN':
            model_SN = model_GCN.GCN_LP(SN,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        n_layers=args.n_layers,
                                        activation=F.relu,
                                        dropout=args.dropout)
        elif args.SN_model == 'GAT':
            heads = [args.num_heads] * (args.n_layers - 1) + [args.num_out_heads]
            model_SN = model_GAT.GAT_LP(SN,
                                        n_layers=args.n_layers,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        heads=heads,
                                        activation=F.elu,
                                        feat_drop=args.in_drop,
                                        attn_drop=args.attn_drop,
                                        negative_slope=args.negative_slope,
                                        residual=args.residual)
        elif args.SN_model == 'GAE':
            model_SN = model_GAE.GAE_LP(SN,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=args.out_dim,
                                        device=device)
        else:
            assert (args.SN_model in ['GCN', 'GAT', 'GAE'])
        model_SN = model_SN.to(device)
        print(SN)
        return SN, model_SN, feature_SN, val_edges, val_edges_false, \
               test_edges, test_edges_false, SN_forward, SN_backward

if __name__ == '__main__':
    if args.task == 'KG_NC':
        KG, model_KG, category_KG, labels_KG, train_idx_KG, valid_idx_KG, test_idx_KG, KG_forward, KG_backward = KG_data_prepare()
        SN, model_SN, feature_SN, val_edges_SN, val_edges_false_SN, test_edges_SN, test_edges_false_SN, SN_forward, SN_backward = SN_data_prepare()

        crition_KG = torch.nn.BCEWithLogitsLoss()
        adj_SN = SN.adjacency_matrix().to_dense().to(device)

        weight_tensor_SN, norm_SN = data_process_SN.compute_loss_para_LP(adj_SN, device)
        model = Model_KG_NC(model_KG, model_SN, args.in_dim, args.out_dim).to(device)
    elif args.task in ['KG_LP', 'SN_LP', 'Align']:
        KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, test_edges_false_KG, KG_forward, KG_backward = KG_data_prepare()
        SN, model_SN, feature_SN, val_edges_SN, val_edges_false_SN, test_edges_SN, test_edges_false_SN, SN_forward, SN_backward = SN_data_prepare()

        adj_SN = SN.adjacency_matrix().to_dense().to(device)
        weight_tensor_SN, norm_SN = data_process_SN.compute_loss_para_LP(adj_SN, device)
        model = Model_KG_LP(model_KG, model_SN, args.in_dim, args.out_dim).to(device)
    elif args.task == 'SN_NC':
        pass
    else:
        pass

    # 生成实体对齐的训练集
    node_align_KG_train, node_align_KG_valid, node_align_KG_test, \
    node_align_SN_train, node_align_SN_valid, node_align_SN_test = \
        data_process_MAIN.node_align_split(args, KG_forward, KG_backward, SN_forward, SN_backward, device)
    node_align_KG_train = node_align_KG_train.repeat(args.neg_num)
    node_align_SN_train = node_align_SN_train.repeat(args.neg_num)

    # 优化器可以进行选择
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=1e-3, pct_start=0.05)
    early_stopper = data_process_MAIN.EarlyStopMonitor(max_round=args.max_round, min_epoch=args.min_epoch,
                                                       tolerance=args.tolerance)

    for epoch in range(args.epochs):
        t = time.time()
        # 这里是在整张图上做训练
        model.train()

        if args.task == 'KG_NC':
            logits_KG, res_mi_KG, res_local_KG, logits_SN, embed_SN, trans_SN, res_mi_SN, res_local_SN = \
                model(category_KG, SN, feature_SN, adj_SN, args.neg_num, device)
            loss_KG = crition_KG(logits_KG[train_idx_KG], labels_KG[train_idx_KG])
            # 这里返回的logits已经经过sigmoid，GAE使用整张图作为训练样本
            loss_SN = norm_SN * F.binary_cross_entropy(logits_SN.view(-1), adj_SN.view(-1), weight=weight_tensor_SN)
        elif args.task in ['KG_LP', 'SN_LP', 'Align']:
            negative_graph_KG = data_process_KG.construct_negative_graph_LP(KG, args.neg_num, triplet, device)
            # print(feature_SN, model_SN)
            pos_score_KG, neg_score_KG, res_mi_KG, res_local_KG, \
            logits_SN, embed_SN, trans_SN, res_mi_SN, res_local_SN = \
                model(KG, negative_graph_KG, triplet, SN, feature_SN, adj_SN, args.neg_num, device)
            loss_KG = data_process_KG.compute_loss_LP(pos_score_KG, neg_score_KG)
            loss_SN = norm_SN * F.binary_cross_entropy(logits_SN.view(-1), adj_SN.view(-1), weight=weight_tensor_SN)
        elif args.task == 'SN_NC':
            pass
        else:
            assert (args.task in ['KG_NC', 'KG_LP', 'SN_NC', 'SN_LP', 'Align'])

        # 计算互信息损失
        res_mi_KG_pos, res_mi_KG_neg = res_mi_KG
        res_local_KG_pos, res_local_KG_neg = res_local_KG
        res_mi_SN_pos, res_mi_SN_neg = res_mi_SN
        res_local_SN_pos, res_local_SN_neg = res_local_SN
        loss_KG_MI = args.alpha * process_GMI.mi_loss_jsd(res_mi_KG_pos, res_mi_KG_neg) + \
                     args.beta * process_GMI.mi_loss_jsd(res_local_KG_pos, res_local_KG_neg)
        loss_SN_MI = args.alpha * process_GMI.mi_loss_jsd(res_mi_SN_pos, res_mi_SN_neg) + \
                     args.beta * process_GMI.mi_loss_jsd(res_local_SN_pos, res_local_SN_neg)

        # 实体对齐损失函数
        # 损失函数增加负采样，这里的负样本从社交网络中选取，因为社交网络的作者节点多
        loss_align = 0
        node_align_KG_neg = node_align_KG_train
        node_align_SN_neg = torch.randint(0, SN.num_nodes(), (len(node_align_SN_train),)).to(device)
        if args.dataset == 'OAG':
            align_target = 'author'
        else:
            align_target = 'person'
        if args.align_dist == 'L2':
            # 向量间的距离作为损失函数
            metrix_pos = KG.nodes[align_target].data['h'][node_align_KG_train] - trans_SN[node_align_SN_train]
            metrix_neg = KG.nodes[align_target].data['h'][node_align_KG_neg] - trans_SN[node_align_SN_neg]
            res_pos = torch.sqrt(torch.sum(metrix_pos * metrix_pos, dim=1))
            res_neg = torch.sqrt(torch.sum(metrix_neg * metrix_neg, dim=1))
            results = args.margin + res_pos - res_neg
            for result in results:
                loss_align += max(0, result)
        elif args.align_dist == 'L1':
            metrix_pos = KG.nodes[align_target].data['h'][node_align_KG_train] - trans_SN[node_align_SN_train]
            metrix_neg = KG.nodes[align_target].data['h'][node_align_KG_neg] - trans_SN[node_align_SN_neg]
            res_pos = torch.sum(torch.abs(metrix_pos), dim=1)
            res_neg = torch.sum(torch.abs(metrix_neg), dim=1)
            results = args.margin + res_pos - res_neg
            for result in results:
                loss_align += max(0, result)
        elif args.align_dist == 'cos':
            # dim=1，计算行向量的相似度
            res_pos = torch.cosine_similarity(KG.nodes[align_target].data['h'][node_align_KG_train], trans_SN[node_align_SN_train], dim=1)
            res_neg = torch.cosine_similarity(KG.nodes[align_target].data['h'][node_align_KG_neg], trans_SN[node_align_SN_neg], dim=1)
            # 余弦相似度在[-1, 1]间，为1相似度高，损失函数就小
            results = args.margin - res_pos + res_neg
            for result in results:
                loss_align += max(0, result)
        loss_align /= len(node_align_KG_train)

        loss = args.w_KG*loss_KG + args.w_SN*loss_SN + args.w_KG_MI*loss_KG_MI + args.w_SN_MI*loss_SN_MI + args.w_align*loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if args.task == 'KG_NC':
                # 知识图谱节点分类的结果
                val_micro_f1_KG, val_macro_f1_KG, _, _ = data_process_KG.get_score_NC(logits_KG, labels_KG, valid_idx_KG)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                      "val_micro_f1_KG=", "{:.5f}".format(val_micro_f1_KG),
                      "val_macro_f1_KG=", "{:.5f}".format(val_macro_f1_KG),
                      )
                if early_stopper.early_stop_check(val_micro_f1_KG, model):
                    print('Use micro_f1_KG as early stopping target')
                    print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    model = early_stopper.model
                    break
            elif args.task == 'KG_LP':
                val_roc_KG, val_ap_KG = data_process_KG.get_score_LP(KG, val_edges_KG, val_edges_false_KG, triplet)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                      "val_AUC_KG=", "{:.5f}".format(val_roc_KG),
                      "val_AP_KG=", "{:.5f}".format(val_ap_KG),
                      )
                if early_stopper.early_stop_check(val_roc_KG, model):
                    print('Use roc_KG as early stopping target')
                    print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    model = early_stopper.model
                    break
            elif args.task == 'SN_NC':
                pass
            elif args.task == 'SN_LP':
                # 社交网络链接预测的结果
                val_roc_SN, val_ap_SN = data_process_SN.get_scores_LP(val_edges_SN, val_edges_false_SN, logits_SN)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                      "val_AUC_SN=", "{:.5f}".format(val_roc_SN),
                      "val_AP_SN=", "{:.5f}".format(val_ap_SN),
                      )
                if early_stopper.early_stop_check(val_roc_SN, model):
                    print('Use roc_SN as early stopping target')
                    print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    model = early_stopper.model
                    break
            elif args.task == 'Align':
                # 实体对齐的指标MRR，hits@10
                # 知识图谱的节点向量KG.nodes[align_target].data['h'][node_align_KG_train]
                # 社交网络的节点向量trans_SN[node_align_SN_train]
                val_MRR_align, val_hits_align = data_process_MAIN.align_scores(
                    KG.nodes[align_target].data['h'],
                    trans_SN,
                    node_align_KG_valid,
                    node_align_SN_valid,
                    sample_num=args.align_sample,
                    hit_pos=args.hits_num,
                    align_dist=args.align_dist)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                      'val_MRR_align=', '{:.5f}'.format(val_MRR_align),
                      "val_hits@{}_align=".format(args.hits_num), '{:.5f}'.format(val_hits_align)
                      )
                # early stopping
                if early_stopper.early_stop_check(val_hits_align, model):
                    print('Use hits@{} as early stopping target'.format(args.hits_num))
                    print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    model = early_stopper.model
                    break

    # 测试
    if args.task == 'KG_NC':
        logits_KG, res_mi_KG, res_local_KG, logits, features, trans_SN, res_mi_SN, res_local_SN = \
            model(category_KG, SN, feature_SN, adj_SN, args.neg_num, device)
        test_micro_f1_KG, test_macro_f1_KG, _, _ = data_process_KG.get_score_NC(logits_KG, labels_KG, test_idx_KG)
        print("test_micro_f1_KG=", "{:.5f}".format(test_micro_f1_KG),
              "test_macro_f1_KG=", "{:.5f}".format(test_macro_f1_KG))
    elif args.task == 'KG_LP':
        test_roc_KG, test_ap_KG = data_process_KG.get_score_LP(KG, test_edges_KG, test_edges_false_KG, triplet)
        print("test_roc_SN=", "{:.5f}".format(test_roc_KG),
              "test_ap_SN=", "{:.5f}".format(test_ap_KG))
    elif args.task == 'SN_NC':
        pass
    elif args.task == 'SN_LP':
        test_roc_SN, test_ap_SN = data_process_SN.get_scores_LP(test_edges_SN, test_edges_false_SN, logits_SN)
        print("test_roc_SN=", "{:.5f}".format(test_roc_SN),
              "test_ap_SN=", "{:.5f}".format(test_ap_SN))
    elif args.task == 'Align':
        test_MRR_align, test_hits_align = data_process_MAIN.align_scores(KG.nodes[align_target].data['h'],
                                                                          trans_SN,
                                                                          node_align_KG_valid,
                                                                          node_align_SN_valid,
                                                                          sample_num=args.align_sample,
                                                                          hit_pos=args.hits_num,
                                                                          align_dist=args.align_dist)
        print('test_MRR_align=', '{:.5f}'.format(test_MRR_align),
              "test_hits@{}_align=".format(args.hits_num), '{:.5f}'.format(test_hits_align))
    print(args)