import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import sys
sys.path.append('../MAIN')
import data_process_MAIN
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import model_GCN
import data_process_GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main_NC(args):
    # load and preprocess dataset
    # if args.dataset == 'cora':
    #     data = CoraGraphDataset()
    # elif args.dataset == 'citeseer':
    #     data = CiteseerGraphDataset()
    # elif args.dataset == 'pubmed':
    #     data = PubmedGraphDataset()
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))
    # print(data[0])
    #
    #
    # g = data[0]

    g, _, _ = data_process_MAIN.OAG_SN_ReadData_NC(args)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    features = g.ndata['feature']

    # 这个是做节点分类的，所以mask在节点上
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['valid_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = labels.shape[1]
    # exit()
    # g = OAG_SN_ReadData()
    # features = g.ndata['feature']
    # in_feats = features.shape[1]

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = model_GCN.GCN_NC(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.CrossEntropyLoss()
    crition_KG = torch.nn.BCEWithLogitsLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits, _, _ = model(features)
        loss = crition_KG(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

            # 多标签分类评价指标
            logits = torch.sigmoid(logits)
            label = labels[train_mask].cpu().detach().numpy().reshape(1, -1)[0]
            pred = logits[train_mask].cpu().detach().numpy().reshape(1, -1)[0]
            roc_score = roc_auc_score(label, pred)
            ap_score = average_precision_score(labels[train_mask].cpu().detach().numpy(),
                                               logits[train_mask].cpu().detach().numpy())
            # f1_score的输入数据应该全都是0/1，不能有小数
            micro_f1_train = f1_score(labels[train_mask].cpu().detach().numpy(),
                                      logits[train_mask].cpu().detach().numpy() > 0.5, average="micro")
            macro_f1_train = f1_score(labels[train_mask].cpu().detach().numpy(),
                                      logits[train_mask].cpu().detach().numpy() > 0.5, average="macro")
            print('train:', roc_score, ap_score, micro_f1_train, macro_f1_train)

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

# load and preprocess dataset
def main_LP(args):
    g, _, _ = data_process_MAIN.OAG_SN_ReadData_LP()
    # add self loop，因为图中包含入度为0的节点
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    if args.gpu < 0:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = True
        device = torch.device("cuda:{}".format(args.gpu))
    g = g.to(device)

    features = g.ndata['feature']
    in_feats = features.shape[1]

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    adj = g.adjacency_matrix().to_dense().to(device)
    weight_tensor, norm = data_process_GCN.compute_loss_para(adj, device)
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = data_process_GCN.mask_test_edges_dgl(g, adj, args.train_ratio, args.valid_ratio)
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_SN = dgl.edge_subgraph(g, train_edge_idx, preserve_nodes=True).to(device)
    train_SN = train_SN.to(device)
    print(train_SN)

    # create GCN model
    model = model_GCN.GCN_LP(train_SN,
                   in_feats,
                   args.n_hidden,
                   args.n_layers,
                   F.relu,
                   args.dropout)
    model = model.to(device)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        t = time.time()
        model.train()

        # forward
        logits, embeds = model(features)
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = data_process_GCN.get_acc(logits, adj)

        val_roc, val_ap = data_process_GCN.get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self_loop", action='store_true', default=True,
                        help="graph self-loop (default=False)")
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train set ratio.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Valid set ratio.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main_NC(args)