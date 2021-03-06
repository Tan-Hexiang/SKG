import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os
import dgl
import numpy as np

def mask_test_edges_LP(graph, adj, train_ratio=0.7, valid_ratio=0.1):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    num_test = int(np.floor(edges_all.shape[0] * (1-train_ratio-valid_ratio)))
    num_val = int(np.floor(edges_all.shape[0] * valid_ratio))

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
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # if ismember([idx_i, idx_j], test_edges):
        #     continue
        # if ismember([idx_j, idx_i], test_edges):
        #     continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    # ?????????????????????????????????????????????
    # assert ~ismember(val_edges_false, edges_all)
    # ??????????????????????????????????????????GAE???????????????????????????????????????val_edge?????????train_edges?????????
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false

def compute_loss_para_LP(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def get_scores_LP(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        # preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(adj_rec[e[0], e[1]].item())

    preds_neg = []
    for e in edges_neg:
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        preds_neg.append(adj_rec[e[0], e[1]].data)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc_LP(adj_rec, adj_label):
    # ?????????????????????????????????preds????????????????????????????????????????????????labels_all.size(0)??????????????????????????????????????????????????????
    labels_all = adj_label.view(-1).long()
    # ??????
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return float(accuracy)

def get_score_NC(logits, labels, idx):
    logits = torch.sigmoid(logits)
    label = labels[idx].cpu().detach().numpy().reshape(1, -1)[0]
    pred = logits[idx].cpu().detach().numpy().reshape(1, -1)[0]
    roc_score = roc_auc_score(label, pred)
    ap_score = average_precision_score(labels[idx].cpu().detach().numpy(),
                                       logits[idx].cpu().detach().numpy())
    # f1_score??????????????????????????????0/1??????????????????
    micro_f1 = f1_score(labels[idx].cpu().detach().numpy(),
                              logits[idx].cpu().detach().numpy() > 0.5, average="micro")
    macro_f1 = f1_score(labels[idx].cpu().detach().numpy(),
                              logits[idx].cpu().detach().numpy() > 0.5, average="macro")
    return micro_f1, macro_f1, roc_score, ap_score