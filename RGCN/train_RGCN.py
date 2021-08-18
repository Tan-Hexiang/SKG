"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import model_RGCN

import data_process_RGCN
import sys
sys.path.append('../MAIN')
import data_process_MAIN
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def main(args):
    # load graph data
    # if args.dataset == 'aifb':
    #     dataset = AIFBDataset()
    # elif args.dataset == 'mutag':
    #     dataset = MUTAGDataset()
    # elif args.dataset == 'bgs':
    #     dataset = BGSDataset()
    # elif args.dataset == 'am':
    #     dataset = AMDataset()
    # else:
    #     raise ValueError()
    #
    # g = dataset[0]
    # category = dataset.predict_category
    # num_classes = dataset.num_classes
    KG, _, _ = data_process_MAIN.OAG_KG_ReadData_NC(args)
    category = 'author'
    train_mask = KG.nodes[category].data.pop('train_mask')
    test_mask = KG.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    labels = KG.nodes[category].data.pop('label')
    num_classes = labels.shape[1]

    # check cuda
    use_cuda = args.cuda >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.cuda)
        KG = KG.to('cuda:%d' % args.cuda)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = model_RGCN.RGCN_NC(KG,
                           32,
                           args.n_hidden,
                           num_classes,
                           # num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model()[category]
        # RuntimeError: bool value of Tensor with more than one value is ambiguous
        crition = torch.nn.BCEWithLogitsLoss()
        loss = crition(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)

        # 多标签分类评价指标
        logits = torch.sigmoid(logits)
        label = labels[train_idx].cpu().detach().numpy().reshape(1,-1)[0]
        pred = logits[train_idx].cpu().detach().numpy().reshape(1,-1)[0]
        roc_score = roc_auc_score(label, pred)
        ap_score = average_precision_score(labels[train_idx].cpu().detach().numpy(), logits[train_idx].cpu().detach().numpy())
        # f1_score的输入数据应该全都是0/1，不能有小数
        micro_f1_train = f1_score(labels[train_idx].cpu().detach().numpy(),
                                  logits[train_idx].cpu().detach().numpy()>0.5, average="micro")
        macro_f1_train = f1_score(labels[train_idx].cpu().detach().numpy(),
                                  logits[train_idx].cpu().detach().numpy() > 0.5, average="macro")
        print('train:', roc_score, ap_score, micro_f1_train, macro_f1_train)
        # micro_f1_train = metrics.f1_score(Y_test, Y_pred, average="micro")
        # train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        # val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        # val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        # print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
        #       format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    if args.model_path is not None:
        torch.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward()[category]
    logits = torch.sigmoid(logits)
    label = labels[test_idx].cpu().detach().numpy().reshape(1, -1)[0]
    pred = logits[test_idx].cpu().detach().numpy().reshape(1, -1)[0]
    roc_score = roc_auc_score(label, pred)
    ap_score = average_precision_score(label, pred)
    micro_f1_test = f1_score(labels[train_idx].cpu().detach().numpy(),
                              logits[train_idx].cpu().detach().numpy() > 0.5, average="micro")
    macro_f1_test = f1_score(labels[train_idx].cpu().detach().numpy(),
                              logits[train_idx].cpu().detach().numpy() > 0.5, average="macro")
    print('test:', roc_score, ap_score, micro_f1_test, macro_f1_test)

def main_LP(args):
    # load graph data
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")
    KG, _, _ = data_process_MAIN.OAG_KG_ReadData_LP()
    triplet = ('author', 'study', 'field')
    adj_orig = KG.adjacency_matrix(etype=triplet[1]).to_dense()
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = \
        data_process_RGCN.mask_test_edges_dgl(KG, adj_orig, triplet[1], args.train_ratio, args.valid_ratio)
    train_edge_idx = torch.tensor(train_edge_idx)

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

    KG = KG.to(device)
    # model_KG = RGCN_LP(KG, args.dim_init, args.hidden1, args.dim_embed, n_layers=2, n_heads=4, use_norm=True).to(device)

    # create model
    model = model_RGCN.RGCN_LP(KG, args.dim_init, args.n_hidden,args.out_dim,
                    num_hidden_layers=args.n_layers, dropout=args.dropout, use_self_loop=args.use_self_loop)
    model_KG = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=1000, max_lr=1e-3, pct_start=0.05)

    # training loop
    print("start training...")
    for epoch in range(1000):
        # 这里是在整张图上做训练
        model_KG.train()
        negative_graph = data_process_RGCN.construct_negative_graph(KG, 1, triplet, device)
        pos_score, neg_score = model_KG(negative_graph, triplet)
        loss = data_process_RGCN.compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc = data_process_RGCN.get_acc(pos_score, neg_score)
        val_roc, val_ap = data_process_RGCN.get_score(KG, val_edges, val_edges_false, triplet)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))
        continue
        # print('Valid Accuracy:', get_score())
        # print("Epoch:", '%04d' % (epoch + 1), 'Loss:', loss.item(), "Train Accuracy:", train_acc)

    test_roc, test_ap = data_process_RGCN.get_score(KG, test_edges, test_edges_false, triplet)
    print("test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dataset", type=str, default='aifb',
                        help="dataset to use")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--dim_init", type=int, default=768,
                        help="number of hidden units")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden units")
    parser.add_argument("--out_dim", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--cuda", type=int, default=0,
            help="cuda")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio.')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Valid set ratio.')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)