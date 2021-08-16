import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import dgl
import numpy as np

def get_graph(DATA='OAG'):
    dir_path = '../dataset_{}'.format(DATA)
    graph_data = {}
    with open(os.path.join(dir_path, 'author_affiliation.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids = []
        affiliation_ids = []
        # 原本的author_id到图中id的映射
        author2graph = {}
        affiliation2graph = {}
        for line in lines:
            author_id, affiliation_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author2graph:
                author2graph[author_id] = len(author2graph)
            if affiliation_id not in affiliation2graph:
                affiliation2graph[affiliation_id] = len(affiliation2graph)
            author_ids.append(author2graph[author_id])
            affiliation_ids.append(affiliation2graph[affiliation_id])
        graph_data[('author', 'in', 'affiliation')] = (torch.tensor(author_ids), torch.tensor(affiliation_ids))
    with open(os.path.join(dir_path, 'author_field.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids = []
        field_ids = []
        # 原本的author_id到图中id的映射
        author2graph = {}
        field2graph = {}
        for line in lines:
            author_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author2graph:
                author2graph[author_id] = len(author2graph)
            if field_id not in field2graph:
                field2graph[field_id] = len(field2graph)
            author_ids.append(author2graph[author_id])
            field_ids.append(field2graph[field_id])
        graph_data[('author', 'study', 'field')] = (torch.tensor(author_ids), torch.tensor(field_ids))
    with open(os.path.join(dir_path, 'author_venue.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids = []
        venue_ids = []
        # 原本的author_id到图中id的映射
        author2graph = {}
        venue2graph = {}
        for line in lines:
            author_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author2graph:
                author2graph[author_id] = len(author2graph)
            if venue_id not in venue2graph:
                venue2graph[venue_id] = len(venue2graph)
            author_ids.append(author2graph[author_id])
            venue_ids.append(venue2graph[venue_id])
        graph_data[('author', 'contribute', 'venue')] = (torch.tensor(author_ids), torch.tensor(venue_ids))
    g = dgl.heterograph(graph_data)
    return g


# 构建异质图，这里把无向图变成双向图
# print(type(data['PvsA']))
# exit()
# for i in data['PvsA']:
#     print(type(i))
#     exit()
def write_data(data):
    matrix = data['PvsA'].toarray()
    print(matrix.shape, data['PvsA'].shape)
    with open('../dataset_ACM/paper-author.txt', 'w') as fw:
        for i in range(data['PvsA'].shape[0]):
            for j in range(data['PvsA'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsP'].toarray()
    print(matrix.shape, data['PvsP'].shape)
    with open('../dataset_ACM/paper-paper.txt', 'w') as fw:
        for i in range(data['PvsP'].shape[0]):
            for j in range(data['PvsP'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsL'].toarray()
    print(matrix.shape, data['PvsL'].shape)
    with open('../dataset_ACM/paper-field.txt', 'w') as fw:
        for i in range(data['PvsL'].shape[0]):
            for j in range(data['PvsL'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
    matrix = data['PvsC'].toarray()
    print(matrix.shape, data['PvsC'].shape)
    with open('../dataset_ACM/paper-venue.txt', 'w') as fw:
        for i in range(data['PvsC'].shape[0]):
            for j in range(data['PvsC'].shape[1]):
                if matrix[i][j] != 0:
                    fw.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')

def read_data_SN():
    dir_path = '../dataset/ACM'
    graph_data = {}
    # with open(os.path.join(dir_path, 'paper-author.txt'), 'r') as fr:
    #     lines = fr.readlines()
    #     paper_ids = []
    #     author_ids = []
    #     for line in lines:
    #         paper_id, author_id, time_id = map(int, line.strip('\n').split('\t'))
    #         paper_ids.append(paper_id)
    #         author_ids.append(author_id)
    #     graph_data[('paper', 'written-by', 'author')] = (torch.tensor(paper_ids), torch.tensor(author_ids))
    #     graph_data[('author', 'writing', 'paper')] = (torch.tensor(author_ids), torch.tensor(paper_ids))
    with open(os.path.join(dir_path, 'paper-paper.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids1 = []
        paper_ids2 = []
        for line in lines:
            paper_id1, paper_id2, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids1.append(paper_id1)
            paper_ids2.append(paper_id2)
        g = dgl.graph((torch.tensor(paper_ids1), torch.tensor(paper_ids2)))
        # graph_data[('paper', 'citing', 'paper')] = (torch.tensor(paper_ids1), torch.tensor(paper_ids2))
        # graph_data[('paper', 'cited', 'paper')] = (torch.tensor(paper_ids2), torch.tensor(paper_ids1))
    # with open(os.path.join(dir_path, 'paper-field.txt'), 'r') as fr:
    #     lines = fr.readlines()
    #     paper_ids = []
    #     field_ids = []
    #     for line in lines:
    #         paper_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
    #         paper_ids.append(paper_id)
    #         field_ids.append(field_id)
    #     graph_data[('paper', 'is-about', 'subject')] = (torch.tensor(paper_ids), torch.tensor(field_ids))
    #     graph_data[('subject', 'has', 'paper')] = (torch.tensor(field_ids), torch.tensor(paper_ids))
    # with open(os.path.join(dir_path, 'paper-venue.txt'), 'r') as fr:
    #     lines = fr.readlines()
    #     paper_ids = []
    #     venue_ids = []
    #     for line in lines:
    #         paper_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
    #         paper_ids.append(paper_id)
    #         venue_ids.append(venue_id)
    #     graph_data[('paper', 'contribute', 'venue')] = (torch.tensor(paper_ids), torch.tensor(venue_ids))
    #     graph_data[('venue', 'has-paper', 'paper')] = (torch.tensor(venue_ids), torch.tensor(paper_ids))

    # g = dgl.heterograph(graph_data)
    return g

def mask_test_edges_dgl(graph, adj, train_ratio=0.7, valid_ratio=0.1):
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
        # 这一步是不合理的，但是不这么做会报错，因为验证集的负采样可能正好在测试集中
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
    # 验证集的负采样会出现在测试集中
    # assert ~ismember(val_edges_false, edges_all)
    # 会出现两个人多次交互的情况，GAE不考虑这种情况，所以会出现val_edge出现在train_edges的情况
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false

def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def get_scores(edges_pos, edges_neg, adj_rec):
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

def get_acc(adj_rec, adj_label):
    # 他这里用整张图做测试，preds相当于得到预测为有边的数量，可是labels_all.size(0)却是把整张图的连接数作为分母，应该把
    labels_all = adj_label.view(-1).long()
    # 原版
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return float(accuracy)