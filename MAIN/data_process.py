import os
import torch
import dgl
import pandas as pd
import dill
import numpy as np
import torch.nn as nn

def OAG_KG_ReadData():
    # 读取node_feature
    graph = dill.load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_ML.pk', 'rb'))

    dir_path = '../dataset/OAG'
    graph_data = {}
    # forward表示从dgl的id转成以前的id，backward表示从原本的节点id转成当前的dgl图id
    author_forward, author_backward = {}, {}
    affiliation_forward, affiliation_backward = {}, {}
    venue_forward, venue_backward = {}, {}
    field_forward, field_backward = {}, {}
    # 这里需要将节点的id进行转化，变成从0开始的节点
    with open(os.path.join(dir_path, 'author_affiliation.txt'), 'r') as fr:
        lines = fr.readlines()
        affiliation_ids = []
        author_ids = []
        author_affiliation_time = []
        for line in lines:
            author_id, affiliation_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author_backward:
                author_backward[author_id] = len(author_backward)
                author_forward[len(author_backward)-1] = author_id
            author_ids.append(author_backward[author_id])
            if affiliation_id not in affiliation_backward:
                affiliation_backward[affiliation_id] = len(affiliation_backward)
                affiliation_forward[len(affiliation_backward)-1] = affiliation_id
            affiliation_ids.append(affiliation_backward[affiliation_id])
            author_affiliation_time.append(time_id)
        graph_data[('author', 'in', 'affiliation')] = (torch.tensor(author_ids), torch.tensor(affiliation_ids))
        graph_data[('affiliation', 'has', 'author')] = (torch.tensor(affiliation_ids), torch.tensor(author_ids))
    with open(os.path.join(dir_path, 'author_field.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids = []
        field_ids = []
        author_field_time = []
        for line in lines:
            author_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author_backward:
                author_backward[author_id] = len(author_backward)
                author_forward[len(author_backward) - 1] = author_id
            author_ids.append(author_backward[author_id])
            if field_id not in field_backward:
                field_backward[field_id] = len(field_backward)
                field_forward[len(field_backward) - 1] = field_id
            field_ids.append(field_backward[field_id])
            author_field_time.append(time_id)
        graph_data[('author', 'study', 'field')] = (torch.tensor(author_ids), torch.tensor(field_ids))
        # graph_data[('field', 'be-studied', 'author')] = (torch.tensor(field_ids), torch.tensor(author_ids))

    with open(os.path.join(dir_path, 'author_venue.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids = []
        venue_ids = []
        author_venue_time = []
        for line in lines:
            author_id, venue_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author_backward:
                author_backward[author_id] = len(author_backward)
                author_forward[len(author_backward) - 1] = author_id
            author_ids.append(author_backward[author_id])
            if venue_id not in venue_backward:
                venue_backward[venue_id] = len(venue_backward)
                venue_forward[len(venue_backward) - 1] = venue_id
            venue_ids.append(venue_backward[venue_id])
            author_venue_time.append(time_id)
        graph_data[('author', 'contribute', 'venue')] = (torch.tensor(author_ids), torch.tensor(venue_ids))
        graph_data[('venue', 'be-contributed', 'author')] = (torch.tensor(venue_ids), torch.tensor(author_ids))

    g = dgl.heterograph(graph_data)
    # 设置每条边的时间戳
    g.edges['in'].data['timestamp'] = torch.tensor(author_affiliation_time)
    g.edges['has'].data['timestamp'] = torch.tensor(author_affiliation_time)
    g.edges['study'].data['timestamp'] = torch.tensor(author_field_time)
    # g.edges['be-studied'].data['timestamp'] = torch.tensor(author_field_time)
    g.edges['contribute'].data['timestamp'] = torch.tensor(author_venue_time)
    g.edges['be-contributed'].data['timestamp'] = torch.tensor(author_venue_time)

    # 设置每个节点的特征向量
    # 输出DataFrame的列名
    # print(graph.node_feature['author'].columns.values.tolist())
    # 节点的初始尺寸都是768
    author_features = []
    for author_id in author_backward.keys():
        author_feature = np.array(graph.node_feature['author'].loc[author_id, 'emb'])
        author_features.append(author_feature)
    author_features = torch.tensor(author_features, dtype=torch.float32)
    affiliation_features = []
    for affiliation_id in affiliation_backward.keys():
        affiliation_feature = np.array(graph.node_feature['affiliation'].loc[affiliation_id, 'emb'])
        affiliation_features.append(affiliation_feature)
    affiliation_features = torch.tensor(affiliation_features, dtype=torch.float32)
    field_features = []
    for field_id in field_backward.keys():
        field_feature = np.array(graph.node_feature['field'].loc[field_id, 'emb'])
        field_features.append(field_feature)
    field_features = torch.tensor(field_features, dtype=torch.float32)
    venue_features = []
    for venue_id in venue_backward.keys():
        venue_feature = np.array(graph.node_feature['venue'].loc[venue_id, 'emb'])
        venue_features.append(venue_feature)
    venue_features = torch.tensor(venue_features, dtype=torch.float32)
    g.nodes['author'].data['feature'] = author_features
    g.nodes['affiliation'].data['feature'] = affiliation_features
    g.nodes['field'].data['feature'] = field_features
    g.nodes['venue'].data['feature'] = venue_features
    return g, author_forward, author_backward

def align_scores(embed_KG, trans_SN, node_align_KG_valid, node_align_SN_valid,
                   sample_num=5, align_dist='L2'):
    KG_valid = node_align_KG_valid.cpu().numpy()
    SN_valid = node_align_SN_valid.cpu().numpy()
    # 头尾实体各替换5次
    align_MRR = 0
    align_hits5 = 0
    for e1, e2 in zip(KG_valid, SN_valid):
        # 可能会和e1, e2相同
        e1_c = np.random.choice(KG_valid, sample_num, replace=False)
        e2_c = np.random.choice(SN_valid, sample_num, replace=False)
        e_KG = np.ones(sample_num, dtype=np.int) * e1
        e_SN = np.ones(sample_num, dtype=np.int) * e2
        # 最后一对是正样本
        e_KG = np.append(np.append(e_KG, e1_c), e1)
        e_SN = np.append(np.append(e2_c, e_SN), e2)
        # 计算距离和排名
        if align_dist == 'L2':
            # 向量间的距离作为分数，距离越小分数越高
            metrix = embed_KG[e_KG] - trans_SN[e_SN]
            metrix = metrix * metrix
            result = torch.sum(metrix, dim=1)
            result_sort, indice = torch.sort(result)
            # 因为排位从0开始，所以计算时要加1
            index = (indice == sample_num * 2).nonzero()[0][0] + 1
        elif align_dist == 'L1':
            # 向量间的曼哈顿距离作为分数，距离越小分数越高
            metrix = embed_KG[e_KG] - trans_SN[e_SN]
            metrix = torch.abs(metrix)
            result = torch.sum(metrix, dim=1)
            result_sort, indice = torch.sort(result)
            index = (indice == sample_num * 2).nonzero()[0][0] + 1
        elif align_dist == 'cos':
            # dim=1，计算行向量的相似度
            result = torch.cosine_similarity(embed_KG[e_KG],
                                             trans_SN[e_SN], dim=1)
            # 余弦相似度，值越大分数越高
            result_sort, indice = torch.sort(result, descending=True)
            index = (indice == sample_num * 2).nonzero()[0][0] + 1
        # index = torch.float(index)
        align_MRR += 1.0 / index.float()
        align_hits5 += 1 if index <= 5 else 0
    align_hits5 /= len(KG_valid)
    return align_MRR, align_hits5

def OAG_SN_ReadData():
    # 读取node_feature
    graph = dill.load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_ML.pk', 'rb'))

    dir_path = '../dataset/OAG'
    # forward表示从dgl的id转成以前的id，backward表示从原本的节点id转成当前的dgl图id
    author_forward, author_backward = {}, {}

    # 这里需要将节点的id进行转化，变成从0开始的节点
    with open(os.path.join(dir_path, 'author_author.txt'), 'r') as fr:
        lines = fr.readlines()
        author_ids1 = []
        author_ids2 = []
        author_author_time = []
        for line in lines:
            author_id1, author_id2, time_id = map(int, line.strip('\n').split('\t'))
            if author_id1 not in author_backward:
                author_backward[author_id1] = len(author_backward)
                author_forward[len(author_backward) - 1] = author_id1
            author_ids1.append(author_backward[author_id1])
            if author_id2 not in author_backward:
                author_backward[author_id2] = len(author_backward)
                author_forward[len(author_backward) - 1] = author_id2
            author_ids2.append(author_backward[author_id2])
            author_author_time.append(time_id)
        # graph_data[('author', 'co-author', 'author')] = (torch.tensor(author_ids1), torch.tensor(author_ids2))
        # graph_data[('affiliation', 'has', 'author')] = (torch.tensor(affiliation_ids), torch.tensor(author_ids))

    g = dgl.graph((torch.tensor(author_ids1), torch.tensor(author_ids2)))

    # 设置每个节点的特征向量
    # 输出DataFrame的列名
    # print(graph.node_feature['author'].columns.values.tolist())
    # 节点的初始尺寸都是768
    author_features = []
    for author_id in author_backward.keys():
        author_feature = np.array(graph.node_feature['author'].loc[author_id, 'emb'])
        author_features.append(author_feature)
    author_features = torch.tensor(author_features, dtype=torch.float32)
    g.ndata['feature'] = author_features

    # 设置每条边的时间戳
    g.edata['timestamp'] = torch.tensor(author_author_time, dtype=torch.float64)
    g.edata['label'] = torch.zeros(g.num_edges(), dtype=torch.int32)
    emb = nn.Parameter(torch.Tensor(g.num_edges(), 100), requires_grad=False)  # .to(device)
    nn.init.xavier_uniform_(emb)
    # emb = emb.clone().detach()
    emb = torch.tensor(emb, dtype=torch.float32)
    g.edata['feature'] = emb
    return g, author_forward, author_backward

# 判断训练是否收敛，提前终止
class EarlyStopMonitor(object):
    """
    example:
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)
    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
        model.load_state_dict(torch.load(best_checkpoint_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        break
    else:
        torch.save(model.state_dict(), model.get_checkpoint_path(epoch))
    """
    def __init__(self, max_round=20, min_epoch=150, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        # 因为模型训练较慢，前期效果非常不稳定，所以必须强制要求训练一定轮数
        self.min_epoch = min_epoch
        self.num_round = 0

        self.epoch_count = 1
        self.best_epoch = 1

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.model = None

    def early_stop_check(self, curr_val, model=None):
        # 当前的指标是越高越好还是越低越好
        # 如果不是越高越好，就把当前的值乘-1，变成越高越好
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            self.model = model
        # 如果提升达到tolerance，就保存当前结果
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            self.model = model
        else:
            self.num_round += 1
        self.epoch_count += 1

        return self.num_round >= self.max_round and self.epoch_count > self.min_epoch

if __name__ == '__main__':
    KG, KG_forward, KG_backward = OAG_KG_ReadData()
    sample_KG = set(KG_forward.values())
    print(KG)
    SN, SN_forward, SN_backward = OAG_SN_ReadData()
    sample_SN = set(SN_forward.values())
    print(SN)
    # 有8个初始节点没有和SN中其他节点的链接
    print(len(sample_KG & sample_SN))
