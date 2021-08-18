import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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

def read_data():
    dir_path = '../dataset/ACM'
    graph_data = {}
    with open(os.path.join(dir_path, 'paper-author.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        author_ids = []
        for line in lines:
            paper_id, author_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            author_ids.append(author_id)
        graph_data[('paper', 'written-by', 'author')] = (torch.tensor(paper_ids), torch.tensor(author_ids))
        # graph_data[('author', 'writing', 'paper')] = (torch.tensor(author_ids), torch.tensor(paper_ids))
    # with open(os.path.join(dir_path, 'paper-paper.txt'), 'r') as fr:
    #     lines = fr.readlines()
    #     paper_ids1 = []
    #     paper_ids2 = []
    #     for line in lines:
    #         paper_id1, paper_id2, time_id = map(int, line.strip('\n').split('\t'))
    #         paper_ids1.append(paper_id1)
    #         paper_ids2.append(paper_id2)
    #     graph_data[('paper', 'citing', 'paper')] = (torch.tensor(paper_ids1), torch.tensor(paper_ids2))
    #     graph_data[('paper', 'cited', 'paper')] = (torch.tensor(paper_ids2), torch.tensor(paper_ids1))
    with open(os.path.join(dir_path, 'paper-field.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        field_ids = []
        for line in lines:
            paper_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            field_ids.append(field_id)
        graph_data[('paper', 'is-about', 'subject')] = (torch.tensor(paper_ids), torch.tensor(field_ids))
        graph_data[('subject', 'has', 'paper')] = (torch.tensor(field_ids), torch.tensor(paper_ids))
    with open(os.path.join(dir_path, 'paper-venue.txt'), 'r') as fr:
        lines = fr.readlines()
        paper_ids = []
        venue_ids = []
        for line in lines:
            paper_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
            paper_ids.append(paper_id)
            venue_ids.append(venue_id)
        graph_data[('paper', 'contribute', 'venue')] = (torch.tensor(paper_ids), torch.tensor(venue_ids))
        graph_data[('venue', 'has-paper', 'paper')] = (torch.tensor(venue_ids), torch.tensor(paper_ids))

    g = dgl.heterograph(graph_data)
    return g

def mask_test_edges_dgl(graph, adj, etype, train_ratio=0.7, valid_ratio=0.1):
    src, dst = graph.edges(etype=etype)
    # src是paper，dst是author
    # adj的行数是author，列数是paper
    # print(max(src), max(dst))
    # print(adj.shape)
    # exit()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    # 验证集0.2，测试集0.1
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
    # 因为是异质图，两个节点的类型不一样，所以不用考虑idx_i和idx_j互换的情况
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[1])
        # if idx_i == idx_j:
        #     continue
        # 判断负样本是否出现过
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            # 这里是有向图
            # if ismember([idx_j, idx_i], np.array(test_edges_false)):
            #     continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[1])
        # if idx_i == idx_j:
        #     continue
        # 验证集的负样本只要不在训练集和验证集中出现就可以
        if ismember([idx_i, idx_j], train_edges):
            continue
        # if ismember([idx_j, idx_i], train_edges):
        #     continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        if val_edges_false:
            # if ismember([idx_j, idx_i], np.array(val_edges_false)):
            #     continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    # print(max(val_edges_false))
        # exit()

    assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false

def get_score_LP(graph, edges_pos, edges_neg, triplet):
    src_type, edge_type, dst_type = triplet
    preds = []
    for src, dst in edges_pos:
        # print(graph.nodes[src_type].data['h'][src].shape)
        src_v = graph.nodes[src_type].data['h'][src]
        dst_v = graph.nodes[dst_type].data['h'][dst]
        res = torch.mul(src_v, dst_v).sum()
        preds.append(torch.sigmoid(res).item())
        # preds.append(torch.sigmoid(torch.mm(graph.nodes[src_type].data['h'][src].transpose(0,1), graph.nodes[dst_type].data['h'][dst])))
    preds_neg = []
    for src, dst in edges_neg:
        # preds_neg.append(torch.sigmoid(torch.mul(graph.nodes[src_type].data['h'][src], graph.nodes[dst_type].data['h'][dst]).sum()))
        src_v = graph.nodes[src_type].data['h'][src]
        dst_v = graph.nodes[dst_type].data['h'][dst]
        res = torch.mul(src_v, dst_v).sum()
        preds_neg.append(torch.sigmoid(res).item())
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # 不使用.item()的话ap_score会报错，’Process finished with exit code 139‘
    # print(labels_all.shape, preds_all.shape)
    ap_score = average_precision_score(labels_all, preds_all)
    # ap_score = 0
    return roc_score, ap_score

def get_score_NC(logits, labels, idx):
    logits = torch.sigmoid(logits)
    label = labels[idx].cpu().detach().numpy().reshape(1, -1)[0]
    pred = logits[idx].cpu().detach().numpy().reshape(1, -1)[0]
    roc_score = roc_auc_score(label, pred)
    ap_score = average_precision_score(labels[idx].cpu().detach().numpy(),
                                       logits[idx].cpu().detach().numpy())
    # f1_score的输入数据应该全都是0/1，不能有小数
    micro_f1 = f1_score(labels[idx].cpu().detach().numpy(),
                              logits[idx].cpu().detach().numpy() > 0.5, average="micro")
    macro_f1 = f1_score(labels[idx].cpu().detach().numpy(),
                              logits[idx].cpu().detach().numpy() > 0.5, average="macro")
    return micro_f1, macro_f1, roc_score, ap_score

def construct_negative_graph(graph, k, etype, device):
    # k表示负样本的比例
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

    # 交叉熵
    # 先把分数转成概率
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(neg_score)
    label = torch.cat((torch.ones(pos_score.shape), torch.zeros(neg_score.shape)), 0)#.to(device)
    pred = torch.cat((pos_score, neg_score), 0)#.to(device)
    loss = F.binary_cross_entropy(pred.view(-1), label.view(-1))
    return loss

def get_acc(pos_score, neg_score):
    # pos_score>0.5的是预测正确的，neg_score<0.5的是正确的
    accuracy = (torch.sigmoid(pos_score)>0.5).sum().float()+(torch.sigmoid(neg_score)<0.5).sum().float()
    accuracy = accuracy / (pos_score.shape[0]+neg_score.shape[0])
    return float(accuracy)