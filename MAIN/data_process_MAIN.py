import os
import torch
import dgl
import pandas as pd
import dill
import numpy as np
import torch.nn as nn

def align_scores(embed_KG, trans_SN, node_align_KG_valid, node_align_SN_valid,
                   sample_num=5, hit_pos=5, align_dist='L2'):
    KG_valid = node_align_KG_valid.cpu().numpy()
    SN_valid = node_align_SN_valid.cpu().numpy()
    # 头尾实体各替换5次
    align_MRR = 0
    align_hits = 0
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
        align_hits += 1 if index <= hit_pos else 0
    align_MRR /= len(KG_valid)
    align_hits /= len(KG_valid)
    return align_MRR, align_hits

def OAG_KG_ReadData_LP(args=None):
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

def OAG_KG_ReadData_NC(args=None):
    # 节点分类预测每个作者的领域
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
        author_field = []
        for line in lines:
            author_id, field_id, time_id = map(int, line.strip('\n').split('\t'))
            if author_id not in author_backward:
                author_backward[author_id] = len(author_backward)
                author_forward[len(author_backward) - 1] = author_id
            if field_id not in field_backward:
                field_backward[field_id] = len(field_backward)
                field_forward[len(field_backward) - 1] = field_id
            author_field.append((author_backward[author_id], field_backward[field_id]))
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
    # g.edges['study'].data['timestamp'] = torch.tensor(author_field_time)
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
    venue_features = []
    for venue_id in venue_backward.keys():
        venue_feature = np.array(graph.node_feature['venue'].loc[venue_id, 'emb'])
        venue_features.append(venue_feature)
    venue_features = torch.tensor(venue_features, dtype=torch.float32)
    g.nodes['author'].data['feature'] = author_features
    g.nodes['affiliation'].data['feature'] = affiliation_features
    # g.nodes['field'].data['feature'] = field_features
    g.nodes['venue'].data['feature'] = venue_features

    # 设置训练标签
    nodes = np.array(list(author_forward.keys()))
    np.random.shuffle(nodes)
    train_idx = int(len(nodes) * args.train_ratio)
    valid_idx = int(len(nodes) * (args.train_ratio + args.valid_ratio))
    node_align_KG_train = nodes[:train_idx]
    node_align_KG_valid = nodes[train_idx:valid_idx]
    node_align_KG_test = nodes[valid_idx:]
    train_mask = torch.zeros(g.num_nodes('author'), dtype=torch.bool)
    valid_mask = torch.zeros(g.num_nodes('author'), dtype=torch.bool)
    test_mask = torch.zeros(g.num_nodes('author'), dtype=torch.bool)
    train_mask[node_align_KG_train] = True
    valid_mask[node_align_KG_valid] = True
    test_mask[node_align_KG_test] = True
    # 测试，或应该全是，与应该全是False
    # print(train_mask | valid_mask | test_mask)
    g.nodes['author'].data['train_mask'] = train_mask
    g.nodes['author'].data['valid_mask'] = valid_mask
    g.nodes['author'].data['test_mask'] = test_mask

    # 设置节点标签
    labels = torch.zeros((g.num_nodes('author'), len(field_forward)), dtype=torch.float32)
    for author_id, field_id in author_field:
        labels[author_id, field_id] = 1
    g.nodes['author'].data['label'] = labels

    return g, author_forward, author_backward

def OAG_SN_ReadData_LP(args=None):
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

def OAG_SN_ReadData_NC(args=None):
    # 生成节点分类的数据集，需要将KG的
    # 读取node_feature
    graph = dill.load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_ML.pk', 'rb'))

    dir_path = '../dataset/OAG'
    # forward表示从dgl的id转成以前的id，backward表示从原本的节点id转成当前的dgl图id
    author_forward, author_backward = {}, {}

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
    author_features = []
    for author_id in author_backward.keys():
        author_feature = np.array(graph.node_feature['author'].loc[author_id, 'emb'])
        author_features.append(author_feature)
    author_features = torch.tensor(author_features, dtype=torch.float32)
    g.ndata['feature'] = author_features


    # 给SN中的节点打标签，需要使用KG的信息
    SN_forward, SN_backward = author_forward, author_backward
    KG, KG_forward, KG_backward = OAG_KG_ReadData_NC(args)
    nodes = set(KG_forward.values()) & set(SN_forward.values())
    nodes_tmp = [KG_backward[node] for node in nodes]
    # 这里nodes保存的是KG中的节点id
    nodes_KG = np.array(nodes_tmp)
    nodes_SN = [SN_backward[KG_forward[node]] for node in nodes_KG]
    labels = torch.zeros((g.num_nodes(), KG.nodes['author'].data['label'].shape[1]), dtype=torch.float32)
    for node_SN, node_KG in zip(nodes_SN, nodes_KG):
        labels[node_SN] = KG.nodes['author'].data['label'][node_KG]
        # print(labels[node_SN])
        # print(KG.nodes['author'].data['label'][node_KG])
    g.ndata['label'] = labels

    # 划分数据集
    np.random.shuffle(nodes_SN)
    train_idx = int(len(nodes_SN) * args.train_ratio)
    valid_idx = int(len(nodes_SN) * (args.train_ratio + args.valid_ratio))
    node_SN_train = nodes_SN[:train_idx]
    node_SN_valid = nodes_SN[train_idx:valid_idx]
    node_SN_test = nodes_SN[valid_idx:]
    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    valid_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    train_mask[node_SN_train] = True
    valid_mask[node_SN_valid] = True
    test_mask[node_SN_test] = True

    # print(train_mask | valid_mask | test_mask) # 有True有False，这里的True就是有label的节点，False是没有label的节点
    # print((train_mask | valid_mask | test_mask).sum()) # 应该是对齐节点的数量
    # print(train_mask & valid_mask & test_mask) # 全是False
    g.ndata['train_mask'] = train_mask
    g.ndata['valid_mask'] = valid_mask
    g.ndata['test_mask'] = test_mask

    return g, author_forward, author_backward

def WDT_KG_ReadData_LP(args):
    dir_path = '../dataset/WDT'
    graph_data = {}
    # wiki2id表示wikidata名字对应的id，id2wiki表示id对应的wikidata名字
    # 人物名字，国籍，隶属组织，出生地
    wiki2id, id2wiki = {}, {}
    country2id, id2country = {}, {}
    school2id, id2school = {}, {}
    affiliation2id, id2affiliation = {}, {}
    birth2id, id2birth = {}, {}
    # 这里需要将节点的id进行转化，变成从0开始的节点
    labels = ['actor', 'dancer', 'politician', 'programmer', 'singer']
    # 这里面一个人可能会换国籍，因此会有多个国籍，比如英属印度和印度
    wiki_ids_country = []
    wiki_ids_school = []
    wiki_ids_affiliation = []
    wiki_ids_birth = []
    country_ids = []
    school_ids = []
    affiliation_ids = []
    birth_ids = []
    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'country_of_citizenship.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, country = line.strip('\n').split('\t')
                if wiki not in wiki2id:
                    wiki2id[wiki] = len(wiki2id)
                    id2wiki[len(wiki2id)-1] = wiki
                # else:
                #     print(label_wiki, '\t', wiki)
                wiki_ids_country.append(wiki2id[wiki])
                if country not in country2id:
                    country2id[country] = len(country2id)
                    id2country[len(country2id)-1] = country
                country_ids.append(country2id[country])
        with open(os.path.join(dir_path, label_name, 'educated_at.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, school = line.strip('\n').split('\t')
                if wiki not in wiki2id:
                    wiki2id[wiki] = len(wiki2id)
                    id2wiki[len(wiki2id)-1] = wiki
                wiki_ids_school.append(wiki2id[wiki])
                if school not in school2id:
                    school2id[school] = len(school2id)
                    id2school[len(school2id)-1] = school
                school_ids.append(school2id[school])
        with open(os.path.join(dir_path, label_name, 'employer.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, affiliation = line.strip('\n').split('\t')
                if wiki not in wiki2id:
                    wiki2id[wiki] = len(wiki2id)
                    id2wiki[len(wiki2id)-1] = wiki
                wiki_ids_affiliation.append(wiki2id[wiki])
                if affiliation not in affiliation2id:
                    affiliation2id[affiliation] = len(affiliation2id)
                    id2affiliation[len(affiliation2id)-1] = affiliation
                affiliation_ids.append(affiliation2id[affiliation])
        with open(os.path.join(dir_path, label_name, 'place_of_birth.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, birth = line.strip('\n').split('\t')
                if wiki not in wiki2id:
                    wiki2id[wiki] = len(wiki2id)
                    id2wiki[len(wiki2id)-1] = wiki
                wiki_ids_birth.append(wiki2id[wiki])
                if birth not in birth2id:
                    birth2id[birth] = len(birth2id)
                    id2birth[len(birth2id)-1] = birth
                birth_ids.append(birth2id[birth])
    graph_data[('person', 'in', 'country')] = (torch.tensor(wiki_ids_country), torch.tensor(country_ids))
    graph_data[('country', 'rev_in', 'person')] = (torch.tensor(country_ids), torch.tensor(wiki_ids_country))
    graph_data[('person', 'educate', 'school')] = (torch.tensor(wiki_ids_school), torch.tensor(school_ids))
    graph_data[('school', 'rev_educate', 'person')] = (torch.tensor(school_ids), torch.tensor(wiki_ids_school))
    graph_data[('person', 'employ', 'affiliation')] = (torch.tensor(wiki_ids_affiliation), torch.tensor(affiliation_ids))
    # graph_data[('affiliation', 'rev_employ', 'person')] = (torch.tensor(affiliation_ids), torch.tensor(wiki_ids_affiliation))
    graph_data[('person', 'born', 'birth')] = (torch.tensor(wiki_ids_birth), torch.tensor(birth_ids))
    graph_data[('birth', 'rev_born', 'person')] = (torch.tensor(birth_ids), torch.tensor(wiki_ids_birth))

    KG = dgl.heterograph(graph_data)
    # 随机初始化每个节点的特征向量
    for ntype in KG.ntypes:
        emb = nn.Parameter(torch.Tensor(KG.number_of_nodes(ntype), args.dim_init), requires_grad=False)
        nn.init.xavier_uniform_(emb)
        KG.nodes[ntype].data['feature'] = emb

    # 打标签

    return KG, id2wiki, wiki2id

def WDT_KG_ReadData_NC(args):
    pass

def WDT_SN_ReadData_LP(args):
    """
    暂时没有采样稠密的子图，而是直接使用整张社交网络
    :param args:
    :return:
    """
    dir_path = '../dataset/WDT'
    # wiki2id表示wikidata名字对应的id，id2wiki表示id对应的wikidata名字
    # 人物名字，国籍，隶属组织，出生地
    twitter2id, id2twitter = {}, {}
    # 这里需要将节点的id进行转化，变成从0开始的节点
    labels = ['actor', 'dancer', 'politician', 'programmer', 'singer']
    twitter_ids1, twitter_ids2 = [], []
    # 采样的初始节点
    seeds = []
    # 统计每个节点的度

    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'follower_relation.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                # 文件中的关注关系是后一个人关注前一个人
                twitter_id2, twitter_id1 = line.strip('\n').split('\t')
                if twitter_id1 not in twitter2id:
                    twitter2id[twitter_id1] = len(twitter2id)
                    id2twitter[len(twitter2id) - 1] = twitter_id1
                twitter_ids1.append(twitter2id[twitter_id1])
                if twitter_id2 not in twitter2id:
                    twitter2id[twitter_id2] = len(twitter2id)
                    id2twitter[len(twitter2id) - 1] = twitter_id2
                twitter_ids2.append(twitter2id[twitter_id2])
        # with open(os.path.join(dir_path, label_name, 'twittername.txt'), 'r') as fr:
        #     lines = fr.readlines()
        #     for line in lines
    SN = dgl.graph((torch.tensor(twitter_ids1), torch.tensor(twitter_ids2)))

    # 采样一个稠密子图

    # 生成每个节点的特征向量
    emb = nn.Parameter(torch.Tensor(len(SN.nodes()), args.dim_init), requires_grad=False)
    nn.init.xavier_uniform_(emb)
    SN.ndata['feature'] = emb

    return SN, id2twitter, twitter2id

def WDT_SN_ReadData_NC(args):
    pass

def node_align_split(args, KG_forward, KG_backward, SN_forward, SN_backward, device):
    """
    实体对齐的数据集划分
    :param args:
    :param KG_forward:
    :param KG_backward:
    :param SN_forward:
    :param SN_backward:
    :param device:
    :return:
    """
    if args.task == 'Align':
        # 只有这种情况需要划分数据集，其余任务整个数据集都是训练集
        if args.dataset == 'WDT':
            # forward是id2name，backward是name2id
            dir_path = '../dataset/WDT'
            labels = ['actor', 'dancer', 'politician', 'programmer', 'singer']
            nodes = []
            wiki2twitter = {}
            for label_name in labels:
                with open(os.path.join(dir_path, label_name, 'twitter_and_wiki.txt'), 'r') as fr:
                    lines = fr.readlines()
                    for line in lines:
                        wiki, twitter = line.strip('\n').split('\t')
                        # 这里会出现一个实体对应多个社交帐号的情况，只保留最早出现的一个
                        if wiki in KG_backward and twitter in SN_backward and KG_backward[wiki] not in wiki2twitter:
                            wiki2twitter[KG_backward[wiki]] = SN_backward[twitter]
                            nodes.append(KG_backward[wiki])
            np.random.shuffle(nodes)
            # OAG中实体对齐的id需要进行转化，因为两张图的id排列是不一样的
            # 因为有些KG中的节点不一定在SN中，有可能是因为引用数高，但是合作少，所以没有和SN中的节点产生链接
            # 从锚节点中选取训练样本数
            # node_align_KG_train = np.random.choice(nodes, int(len(nodes)*args.train_ratio), replace=False)
            train_idx = int(len(nodes)*args.train_ratio)
            valid_idx = int(len(nodes)*(args.train_ratio+args.valid_ratio))
            node_align_KG_train = nodes[:train_idx]
            node_align_KG_valid = nodes[train_idx:valid_idx]
            node_align_KG_test  = nodes[valid_idx:]
            node_align_SN_train = [wiki2twitter[node] for node in node_align_KG_train]
            node_align_SN_valid = [wiki2twitter[node] for node in node_align_KG_valid]
            node_align_SN_test  = [wiki2twitter[node] for node in node_align_KG_test]
        elif args.dataset == 'OAG':
            nodes = set(KG_forward.values()) & set(SN_forward.values())
            nodes_tmp = [KG_backward[node] for node in nodes]
            # 这里nodes保存的是KG中的节点id
            nodes = np.array(nodes_tmp)
            np.random.shuffle(nodes)
            train_idx = int(len(nodes) * args.train_ratio)
            valid_idx = int(len(nodes) * (args.train_ratio + args.valid_ratio))
            node_align_KG_train = nodes[:train_idx]
            node_align_KG_valid = nodes[train_idx:valid_idx]
            node_align_KG_test = nodes[valid_idx:]
            node_align_SN_train = [SN_backward[KG_forward[node]] for node in node_align_KG_train]
            node_align_SN_valid = [SN_backward[KG_forward[node]] for node in node_align_KG_valid]
            node_align_SN_test = [SN_backward[KG_forward[node]] for node in node_align_KG_test]
        else:
            assert (args.dataset in ['OAG', 'WDT'])

        # CPU->GPU
        node_align_KG_train = torch.from_numpy(node_align_KG_train).to(device)
        node_align_KG_valid = torch.from_numpy(node_align_KG_valid).to(device)
        node_align_KG_test = torch.from_numpy(node_align_KG_test).to(device)
        node_align_SN_train = torch.tensor(node_align_SN_train).to(device)
        node_align_SN_valid = torch.tensor(node_align_SN_valid).to(device)
        node_align_SN_test = torch.tensor(node_align_SN_test).to(device)
    else:
        if args.dataset == 'WDT':
            dir_path = '../dataset/WDT'
            labels = ['actor', 'dancer', 'politician', 'programmer', 'singer']
            nodes = []
            wiki2twitter = {}
            for label_name in labels:
                with open(os.path.join(dir_path, label_name, 'twitter_and_wiki.txt'), 'r') as fr:
                    lines = fr.readlines()
                    for line in lines:
                        wiki, twitter = line.strip('\n').split('\t')
                        # 这里会出现一个实体对应多个社交帐号的情况，只保留最早出现的一个
                        if wiki in KG_backward and twitter in SN_backward and KG_backward[wiki] not in wiki2twitter:
                            wiki2twitter[KG_backward[wiki]] = SN_backward[twitter]
                            nodes.append(KG_backward[wiki])
            node_align_KG_train = nodes
            node_align_SN_train = [wiki2twitter[node] for node in node_align_KG_train]
        elif args.dataset == 'OAG':
            nodes = set(KG_forward.values()) & set(SN_forward.values())
            nodes_tmp = [KG_backward[node] for node in nodes]
            # 这里nodes保存的是KG中的节点id
            nodes = np.array(nodes_tmp)
            node_align_KG_train = nodes
            node_align_SN_train = [SN_backward[KG_forward[node]] for node in node_align_KG_train]
        else:
            assert (args.dataset in ['OAG', 'WDT'])
            # CPU->GPU
        node_align_KG_train = torch.from_numpy(node_align_KG_train).to(device)
        node_align_KG_valid = None
        node_align_KG_test = None
        node_align_SN_train = torch.tensor(node_align_SN_train).to(device)
        node_align_SN_valid = None
        node_align_SN_test = None
    return node_align_KG_train, node_align_KG_valid, node_align_KG_test, \
           node_align_SN_train, node_align_SN_valid, node_align_SN_test

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
    # KG, KG_forward, KG_backward = OAG_KG_ReadData()
    # sample_KG = set(KG_forward.values())
    # print(KG)
    # SN, SN_forward, SN_backward = OAG_SN_ReadData()
    # sample_SN = set(SN_forward.values())
    # print(SN)
    # # 有8个初始节点没有和SN中其他节点的链接
    # print(len(sample_KG & sample_SN))

    import argparse
    parser = argparse.ArgumentParser(description='SKG')
    parser.add_argument('--dim_init', type=int, default=768, help='Dim of initial embedding vector.')
    parser.add_argument('--cuda', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio.')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Valid set ratio.')
    args = parser.parse_args()
    # 测试WDT数据集读取
    # KG, KG_forward, KG_backward = WDT_KG_ReadData(args)
    # print(KG)
    # SN, SN_forward, SN_backward = WDT_SN_ReadData(args)
    # print(SN)
    # # 处理对齐问题
    # node_align(args, KG_backward, SN_backward)

    # 测试节点分类的标签和数据集mask
    # OAG_KG_ReadData_NC(args)

    # 测试SN节点分类数据
    OAG_SN_ReadData_NC(args)
