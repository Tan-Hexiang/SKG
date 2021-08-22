# SN_NC时，KG中应该把一类边去掉，这里没有这么处理，所以会发生信息泄露，不过链接预测不是主任务，所以影响不大
from parser import *
import sys
sys.path.extend(['../RGCN', '../HGT-DGL', '../GCN', '../GAT', '../GAE', '../GMI'])

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
assert (args.dataset in ['OAG', 'WDT'])
assert (args.KG_model in ['RGCN', 'HGT', 'None'])
assert (args.KG_model in ['RGCN', 'HGT', 'None'])
assert (args.SN_model in ['GCN', 'GAT', 'GAE', 'None'])
assert (args.task in ['KG_NC', 'KG_LP', 'SN_NC', 'SN_LP', 'Align'])
assert (args.KG_model != 'None' or args.SN_model != 'None')
if args.KG_model == 'None' and 'KG' in args.task:
    assert True
if args.SN_model == 'None' and 'SN' in args.task:
    assert True

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

def KG_data_prepare():
    # if args.KG_model == 'None':
    #     return tuple([None]*9)
    # KG数据读取
    if args.dataset == 'OAG':
        if args.task in ['KG_NC', 'SN_NC']:
            # 数据读取NC和LP的区别在于NC没有field节点，因为field节点被用来打标签了
            KG, KG_forward, KG_backward = data_process_MAIN.OAG_KG_ReadData_NC(args)
            triplet = ('author', 'contribute', 'venue')
        else:
            KG, KG_forward, KG_backward = data_process_MAIN.OAG_KG_ReadData_LP(args)
            triplet = ('author', 'study', 'field')
        category = 'author'
        
    elif args.dataset == 'WDT':
        if args.task in ['KG_NC', 'SN_NC']:
            KG, KG_forward, KG_backward = data_process_MAIN.WDT_KG_ReadData_NC(args)
        else:
            KG, KG_forward, KG_backward = data_process_MAIN.WDT_KG_ReadData_LP(args)
        category = 'person'
        triplet = ('person', 'employ', 'affiliation')

    if args.task == 'KG_NC':
        KG.node_dict = {}
        KG.edge_dict = {}
        # 给每个类型加上id，从0开始
        for ntype in KG.ntypes:
            KG.node_dict[ntype] = len(KG.node_dict)
        for etype in KG.etypes:
            KG.edge_dict[etype] = len(KG.edge_dict)
            # 貌似dgl的图在to(device)后就不能进行更改了
            KG.edges[etype].data['id'] = torch.ones(KG.number_of_edges(etype), dtype=torch.long) * KG.edge_dict[etype]
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
        else:#if args.KG_model == 'HGT':
            model_KG = model_HGT.HGT_NC(KG,
                                        in_dim=args.in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=num_classes,
                                        n_layers=2,
                                        n_heads=4,
                                        use_norm=True)
        # elif args.KG_model == 'None':
        #     return tuple([None]*9)
        model_KG = model_KG.to(device)
        print(KG)
        return KG, model_KG, category, labels, train_idx, valid_idx, test_idx, KG_forward, KG_backward
    else:
        if args.task == 'KG_LP':
            # KG_LP时，KG需要采样，其余时候不需要
            # 这里的邻接矩阵作用只是提供idx的范围，行数是源节点，列数是目标节点
            adj_orig = KG.adjacency_matrix(etype=triplet[1]).to_dense()
            train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = \
                data_process_KG.mask_test_edges_LP(KG, adj_orig, triplet[1], args.train_ratio, args.valid_ratio)
            train_edge_idx = torch.tensor(train_edge_idx)

            if args.dataset == 'OAG':
                train_graph = KG.edge_subgraph({
                    # 把author/paper这类和各种节点都有连边的点放在源节点的位置，猜测是multi_update_all函数只更新尾实体，所以如果尾实体是单独的节点类型，
                    # 会没有添加't'属性
                    ('author', 'study', 'field'): train_edge_idx,
                    ('author', 'in', 'affiliation'): list(range(KG.number_of_edges('in'))),
                    ('affiliation', 'has', 'author'): list(range(KG.number_of_edges('has'))),
                    ('author', 'contribute', 'venue'): list(range(KG.number_of_edges('contribute'))),
                    ('venue', 'be-contributed', 'author'): list(range(KG.number_of_edges('be-contributed'))),
                }, preserve_nodes=True)
            elif args.dataset == 'WDT':
                train_graph = KG.edge_subgraph({
                    # 把author/paper这类和各种节点都有连边的点放在源节点的位置，猜测是multi_update_all函数只更新尾实体，所以如果尾实体是单独的节点类型，
                    # 会没有添加't'属性
                    ('person', 'employ', 'affiliation'): train_edge_idx,
                    ('person', 'in', 'country'): list(range(KG.number_of_edges('in'))),
                    ('country', 'rev_in', 'person'): list(range(KG.number_of_edges('rev_in'))),
                    ('person', 'educate', 'school'): list(range(KG.number_of_edges('educate'))),
                    ('school', 'rev_educate', 'person'): list(range(KG.number_of_edges('rev_educate'))),
                    ('person', 'born', 'birth'): list(range(KG.number_of_edges('born'))),
                    ('birth', 'rev_born', 'person'): list(range(KG.number_of_edges('rev_born')))
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
        else:#elif args.KG_model == 'HGT':
            model_KG = model_HGT.HGT_LP(KG,
                                        in_dim=args.in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=args.hidden_dim,
                                        n_layers=args.n_layers,
                                        n_heads=1,
                                        use_norm=True)

        model_KG = model_KG.to(device)
        print(KG)
        return KG, model_KG, triplet, val_edges, val_edges_false, test_edges, test_edges_false, KG_forward, KG_backward

def SN_data_prepare():
    # if args.SN_model == 'None':
    #     return tuple([None]*9)
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

    SN = dgl.remove_self_loop(SN)
    SN = dgl.add_self_loop(SN)
    print(SN_forward)
    if args.task == 'SN_NC':
        features = SN.ndata['feature'].to(device)
        labels = SN.ndata['label'].to(device)
        train_idx_SN = SN.ndata['train_mask']
        valid_idx_SN = SN.ndata['valid_mask']
        test_idx_SN = SN.ndata['test_mask']
        in_dim = features.shape[1]
        n_classes = labels.shape[1]

        SN = SN.to(device)
        if args.SN_model == 'GCN':
            model_SN = model_GCN.GCN_NC(SN,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=n_classes,
                                        n_layers=args.n_layers,
                                        activation=F.elu,
                                        dropout=args.dropout)
        elif args.SN_model == 'GAT':
            heads = [args.num_heads] * (args.n_layers - 1) + [args.num_out_heads]
            model_SN = model_GAT.GAT_NC(SN,
                                        n_layers=args.n_layers,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=n_classes,
                                        heads=heads,
                                        activation=F.elu,
                                        feat_drop=args.in_drop,
                                        attn_drop=args.attn_drop,
                                        negative_slope=args.negative_slope,
                                        residual=args.residual
                                        )
        elif args.SN_model == 'GAE':
            model_SN = model_GAE.GAE_NC(SN,
                                        in_dim=in_dim,
                                        hidden_dim=args.hidden_dim,
                                        out_dim=n_classes,
                                        device=device)

        model_SN = model_SN.to(device)

        print(SN)
        return SN, model_SN, labels, features, train_idx_SN, valid_idx_SN, test_idx_SN, SN_forward, SN_backward
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
                                        activation=F.elu,
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
        model_SN = model_SN.to(device)
        print(SN)
        return SN, model_SN, feature_SN, val_edges, val_edges_false, \
               test_edges, test_edges_false, SN_forward, SN_backward

if __name__ == '__main__':
    if args.task == 'KG_NC':
        KG, model_KG, category_KG, labels_KG, train_idx_KG, valid_idx_KG, \
        test_idx_KG, KG_forward, KG_backward = KG_data_prepare()
        SN, model_SN, feature_SN, val_edges_SN, val_edges_false_SN, \
        test_edges_SN, test_edges_false_SN, SN_forward, SN_backward = SN_data_prepare()

        if args.dataset == 'OAG':
            # 多标签分类
            crition_KG = torch.nn.BCEWithLogitsLoss()
        elif args.dataset == 'WDT':
            crition_KG = torch.nn.CrossEntropyLoss()

        adj_SN = SN.adjacency_matrix().to_dense().to(device)
        weight_tensor_SN, norm_SN = data_process_SN.compute_loss_para_LP(adj_SN, device)
        model = Model_KG_NC(model_KG, model_SN, args.in_dim, args.hidden_dim).to(device)
    elif args.task in ['KG_LP', 'SN_LP', 'Align']:
        KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, \
        test_edges_false_KG, KG_forward, KG_backward = KG_data_prepare()
        SN, model_SN, feature_SN, val_edges_SN, val_edges_false_SN, \
        test_edges_SN, test_edges_false_SN, SN_forward, SN_backward = SN_data_prepare()

        adj_SN = SN.adjacency_matrix().to_dense().to(device)
        weight_tensor_SN, norm_SN = data_process_SN.compute_loss_para_LP(adj_SN, device)
        model = Model_KG_LP(model_KG, model_SN, args.in_dim, args.hidden_dim).to(device)
    elif args.task == 'SN_NC':
        KG, model_KG, triplet, val_edges_KG, val_edges_false_KG, test_edges_KG, \
        test_edges_false_KG, KG_forward, KG_backward = KG_data_prepare()
        SN, model_SN, labels_SN, feature_SN, train_idx_SN, valid_idx_SN, \
        test_idx_SN, SN_forward, SN_backward = SN_data_prepare()

        if args.dataset == 'OAG':
            crition_SN = torch.nn.BCEWithLogitsLoss()
        elif args.dataset == 'WDT':
            crition_SN = torch.nn.CrossEntropyLoss()
        model = Model_SN_NC(model_KG, model_SN, args.in_dim, args.hidden_dim).to(device)


    # 生成实体对齐的训练集
    node_align_KG_train, node_align_KG_valid, node_align_KG_test, \
    node_align_SN_train, node_align_SN_valid, node_align_SN_test = \
        data_process_MAIN.node_align_split(args, KG_forward, KG_backward, SN_forward, SN_backward, device)
    node_align_KG_train = node_align_KG_train.repeat(args.neg_num)
    node_align_SN_train = node_align_SN_train.repeat(args.neg_num)

    # 优化器可以进行选择
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=1e-3, pct_start=0.05)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    early_stopper = data_process_MAIN.EarlyStopMonitor(max_round=args.max_round, min_epoch=args.min_epoch,
                                                       tolerance=args.tolerance)
    print(SN_forward)

    for epoch in range(args.epochs):
        t = time.time()
        # 这里是在整张图上做训练
        model.train()

        if args.task == 'KG_NC':
            logits_KG, res_mi_KG, res_local_KG, logits_SN, embed_SN, trans_SN, res_mi_SN, res_local_SN = \
                model(category_KG, SN, feature_SN, adj_SN, args.neg_num, device)
            # 多分类的损失函数要求label类型是torch.long，且维度是num*1，维度1的表示第几类
            if args.dataset == 'OAG':
                loss_KG = crition_KG(logits_KG[train_idx_KG], labels_KG[train_idx_KG])
            elif args.dataset == 'WDT':
                loss_KG = crition_KG(logits_KG[train_idx_KG], labels_KG[train_idx_KG].argmax(dim=1))
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
            negative_graph_KG = data_process_KG.construct_negative_graph_LP(KG, args.neg_num, triplet, device)
            pos_score_KG, neg_score_KG, res_mi_KG, res_local_KG, \
            logits_SN, embed_SN, trans_SN, res_mi_SN, res_local_SN = \
                model(KG, negative_graph_KG, triplet, SN, feature_SN, args.neg_num, device)
            loss_KG = data_process_KG.compute_loss_LP(pos_score_KG, neg_score_KG)
            if args.dataset == 'OAG':
                loss_SN = crition_SN(logits_SN[train_idx_SN], labels_SN[train_idx_SN])
            elif args.dataset == 'WDT':
                loss_SN = crition_SN(logits_SN[train_idx_SN], labels_SN[train_idx_SN].argmax(dim=1))




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
        if args.dataset == 'OAG':
            align_target = 'author'
        else:
            align_target = 'person'

        # print(embed_SN[node_align_SN_valid])
        # print(trans_SN[node_align_SN_valid])
        # print(KG.nodes[align_target].data['h'][node_align_KG_valid])
        # 每十轮生成负样本
        # if epoch % 10 == 0:
            # node_align_KG_neg = node_align_KG_train
        node_align_KG_neg = torch.randint(0, KG.num_nodes(align_target), (len(node_align_KG_train),)).to(device)
            # node_align_SN_neg = torch.randint(0, SN.num_nodes(), (len(node_align_SN_train),)).to(device)
        node_align_SN_neg = torch.randint(0, SN.num_nodes(), (len(node_align_SN_train),)).to(device)
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
            metrix_neg1 = KG.nodes[align_target].data['h'][node_align_KG_train] - trans_SN[node_align_SN_neg]
            metrix_neg2 = KG.nodes[align_target].data['h'][node_align_KG_neg] - trans_SN[node_align_SN_train]
            res_pos = torch.sum(torch.abs(metrix_pos), dim=1)
            res_neg1 = torch.sum(torch.abs(metrix_neg1), dim=1)
            res_neg2 = torch.sum(torch.abs(metrix_neg2), dim=1)
            results1 = args.margin + res_pos - res_neg1
            results2 = args.margin + res_pos - res_neg2
            for result in results1:
                loss_align += max(0, result)
            for result in results2:
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

        if args.KG_model == 'None':
            loss = loss_SN
        elif args.SN_model == 'None':
            loss = loss_KG
        elif args.task == 'Align':
            # loss = loss_align
            loss = 2*loss_align + loss_KG+loss_SN+loss_KG_MI+loss_SN_MI
        else:
            loss = args.w_KG*loss_KG + args.w_SN*loss_SN + \
                   args.w_KG_MI*loss_KG_MI + args.w_SN_MI*loss_SN_MI + \
                   args.w_align*loss_align

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
                if early_stopper.early_stop_check(val_macro_f1_KG, model):
                    print('Use macro_f1_KG as early stopping target')
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
                val_micro_f1_SN, val_macro_f1_SN, _, _ = data_process_SN.get_score_NC(logits_SN, labels_SN,
                                                                                      valid_idx_SN)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                      "val_micro_f1_SN=", "{:.5f}".format(val_micro_f1_SN),
                      "val_macro_f1_SN=", "{:.5f}".format(val_macro_f1_SN),
                      )
                if early_stopper.early_stop_check(val_micro_f1_SN, model):
                    print('Use micro_f1_SN as early stopping target')
                    print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                    print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    model = early_stopper.model
                    break
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
                # print(KG.nodes[align_target].data['h'][node_align_KG_valid[0]])
                # print(KG.nodes[align_target].data['h'][node_align_KG_valid[1]])
                # print('\n\n')
                # print(trans_SN[0])
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
                      "val_hits@{}_align=".format(args.hits_num), val_hits_align
                      )
                # early stopping
                if early_stopper.early_stop_check(val_MRR_align.cpu(), model):
                    # print('Use hits@{} as early stopping target'.format(args.hits_num))
                    print('Use MRR as early stopping target')
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
        negative_graph_KG = data_process_KG.construct_negative_graph_LP(KG, args.neg_num, triplet, device)
        _, _, _, _, logits_SN, _, _, _, _ = \
            model(KG, negative_graph_KG, triplet, SN, feature_SN, args.neg_num, device)
        test_micro_f1_SN, test_macro_f1_SN, _, _ = data_process_SN.get_score_NC(logits_SN, labels_SN, test_idx_SN)
        print("test_micro_f1_SN=", "{:.5f}".format(test_micro_f1_SN),
              "test_macro_f1_SN=", "{:.5f}".format(test_macro_f1_SN))
    elif args.task == 'SN_LP':
        test_roc_SN, test_ap_SN = data_process_SN.get_scores_LP(test_edges_SN, test_edges_false_SN, logits_SN)
        print("test_roc_SN=", "{:.5f}".format(test_roc_SN),
              "test_ap_SN=", "{:.5f}".format(test_ap_SN))
    elif args.task == 'Align':
        test_MRR_align, test_hits_align = data_process_MAIN.align_scores(KG.nodes[align_target].data['h'],
                                                                          trans_SN,
                                                                          node_align_KG_test,
                                                                          node_align_SN_test,
                                                                          sample_num=args.align_sample,
                                                                          hit_pos=args.hits_num,
                                                                          align_dist=args.align_dist)
        print('test_MRR_align=', '{:.5f}'.format(test_MRR_align),
              "test_hits@{}_align=".format(args.hits_num), test_hits_align)
    print(args)