from collections import defaultdict
import dill
import sys
import random
import numpy as np
from data import Graph, sample_subgraph, feature_OAG

# 重要人物，即锚节点的数量
famous_num = 600
sample_depth = 4
sample_width = 500
random.seed(100)
# 选取论文数前10的领域
sample_field_num = 10
# 选取论文数前10的会议
sample_venue_num = 10
dir_path = '../dataset/OAG'
# dir_path = '../../../../../data/lzj/OAG'

# class RenameUnpickler(dill.Unpickler):
#     def find_class(self, module, name):
#         renamed_module = module
#         if module == "GPT_GNN.data" or module == 'data':
#             renamed_module = "data"
#         return super(RenameUnpickler, self).find_class(renamed_module, name)
#
# def renamed_load(file_obj):
#     return RenameUnpickler(file_obj).load()
# graph = renamed_load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_CS.pk', 'rb'))

graph = dill.load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_ML.pk', 'rb'))
SN, KG, SN_final = Graph(), Graph(), Graph()
SN.node_feature = graph.node_feature
KG.node_feature = graph.node_feature
SN_final.node_feature = graph.node_feature

# 可以使用作者的引用数或者节点的邻居数
def get_seed_node(author_author, num):
    """
    在作者社交网络上获取交互最多的num个作者
    :param author_author:
    :param num: 初始节点个数
    :return:返回大牛的id列表
    """
    array = []
    for author_id in author_author.keys():
        # 这里使用节点的邻居作为评价指标
        # array.append([author_id, len(author_author[author_id])])
        # 这里使用每个人的引用数作为评价指标
        array.append([author_id, graph.node_feature['author'].loc[author_id, 'citation']])
    sort_array = sorted(array, key=lambda x: x[1], reverse=True)[:num]
    # print(sort_array[0][1])
    return [item[0] for item in sort_array]

def save():
    # print(type(graph.node_feature['author'].loc[sampled_nodes, :]))
    # 保存各类节点的属性，第一列是实际图中保存的id号sampled_author
    # graph.node_feature['author'].loc[sampled_author, :].to_csv(dir_path+'/node_author.txt', index=True, sep='\t', header=True)
    # graph.node_feature['venue'].loc[list(sampled_venue), :].to_csv('../dataset/OAG/node_venue.txt', index=True, sep='\t', header=True)
    # graph.node_feature['field'].loc[list(sampled_field), :].to_csv('../dataset/OAG/node_field.txt', index=True, sep='\t', header=True)
    # graph.node_feature['affiliation'].loc[list(sampled_affiliation), :].to_csv('../dataset/OAG/node_affiliation.txt', index=True, sep='\t', header=True)
    # 保存连边
    # author-author
    with open(dir_path+'/author_author.txt', 'w') as fw:
        for author_id1 in author_author.keys():
            for author_id2, time_id in author_author[author_id1]:
                fw.write(str(author_id1)+'\t'+str(author_id2)+'\t'+str(time_id)+'\n')
    # author-venue
    with open(dir_path+'/author_venue.txt', 'w') as fw:
        for author_id in author_venue.keys():
            for venue_id, time_id in author_venue[author_id].items():
                fw.write(str(author_id)+'\t'+str(venue_id)+'\t'+str(time_id)+'\n')
    # author-field
    with open(dir_path+'/author_field.txt', 'w') as fw:
        for author_id in author_field.keys():
            for field_id, time_id in author_field[author_id].items():
                fw.write(str(author_id)+'\t'+str(field_id)+'\t'+str(time_id)+'\n')
    # author-affiliation
    with open(dir_path+'/author_affiliation.txt', 'w') as fw:
        for author_id in author_affiliation.keys():
            for affiliation_id, time_id in author_affiliation[author_id].items():
                fw.write(str(author_id)+'\t'+str(affiliation_id)+'\t'+str(time_id)+'\n')
    # 以HGT需要的形式保存SN和KG
    # dill.dump(SN_final, open('../dataset/OAG/SN.pk', 'wb'))
    # dill.dump(KG, open('../dataset/OAG/KG.pk', 'wb'))

# 省去paper节点，将author直接和field、venue和affiliation相连
# 和author相连的节点类型 sample_node_types = ['affiliation', 'paper']
paper_relation = ['AP_write_last', 'AP_write_other', 'AP_write_first']
P_V_relation = ['PV_Conference', 'PV_Journal', 'PV_Repository', 'PV_Patent']
P_F_relation = ['PF_in_L1']
# 整合关系
# 保存所有参与这篇论文的作者
paper_author = defaultdict(set)
paper_time = defaultdict(int)

# Social Network
# author-paper-author
# 保存与作者合作的作者id和时间
author_author = defaultdict(lambda : defaultdict(list))
author_time = defaultdict(int)
for relation in paper_relation:
    paper_ids = graph.edge_list['paper']['author'][relation].keys()
    for paper_id in paper_ids:
        author_ids = list(graph.edge_list['paper']['author'][relation][paper_id].keys())
        if author_ids:
            paper_time[paper_id] = graph.edge_list['paper']['author'][relation][paper_id][author_ids[0]]
        paper_author[paper_id] |= set(author_ids)
for paper_id in paper_author.keys():
    for author_id1 in paper_author[paper_id]:
        # 得到作者的时间，用于子图采样
        if author_time[author_id1] < paper_time[paper_id]:
            author_time[author_id1] = paper_time[paper_id]
        for author_id2 in paper_author[paper_id]:
            if author_id1 == author_id2:
                continue
            # author_author[author_id1].append([author_id2, paper_time[paper_id]])
            author_author[author_id1][author_id2].append(paper_time[paper_id])
            author1 = {'id': author_id1, 'type': 'author'}
            author2 = {'id': author_id2, 'type': 'author'}
            SN.add_edge(author1, author2, relation_type='co-author', time=paper_time[paper_id], directed=False)

# 图中的id是相对id，即add_node的顺序，forward是根据原id得到图中的相对id，bacward是根据图中的相对id得到原id
# 这里使用作者的引用数作为选取seeds的标准
seeds = get_seed_node(author_author, famous_num)
print(seeds)
# print(SN.edge_list['author']['author']['co-author'][SN.node_forward['author'][seeds[0]]])
time_range  = {t: True for t in graph.times if t != None}
target_info = []
for author_id in seeds:
    # 在SN上要先转化成相对id
    target_info.append([SN.node_forward['author'][author_id], author_time[author_id]])
# 给定作者以及作者对应的时间，开始采样
# sampled_depth——采样深度，相当于几跳的邻居
# sampled_number——每层的采样数量，最终得到的节点数是种子节点数+sampled_depth*sampled_number
subgraph = sample_subgraph(SN, time_range, sampled_depth = sample_depth, sampled_number = sample_width, inp = {'author': np.array(target_info)},
                           feature_extractor=feature_OAG)
# 得到最终采样的节点
sampled_author = subgraph[3]['author']
print('SN sampled nodes:', sampled_author, 'length:', len(sampled_author))

# print('author-author:', author_author)
# 得到最终的采样节点后需要重新构建SN
tmp_authot_author = defaultdict(list)
for author_id1 in author_author.keys():
    if author_id1 not in sampled_author:
        continue
    for author_id2 in author_author[author_id1].keys():
        if author_id2 not in sampled_author:
            continue
        for time_id in author_author[author_id1][author_id2]:
            tmp_authot_author[author_id1].append([author_id2, time_id])
        time_id = max(author_author[author_id1][author_id2])
        # if SN_final.edge_list['author']['author']['co-author'][author_id1][author_id2] < time_id:
        # Graph的数据结构无法保存节点间的多次交互，因此只能保存最近的一次
        author1 = {'id': author_id1, 'type': 'author'}
        author2 = {'id': author_id2, 'type': 'author'}
        SN_final.add_edge(author1, author2, relation_type='co-author', time=time_id, directed=False)
author_author = tmp_authot_author
# print('author-author:', author_author)
# print(SN_final.edge_list['author']['author']['co-author'])
print('author-author')

# author-paper-venue
# 根据论文数选择会议集合
# 根据论文数选取领域集
venue_paper = []
for relation in P_V_relation:
    venue_ids = list(graph.edge_list['venue']['paper'][relation].keys())
    for venue_id in venue_ids:
        paper_num = len(graph.edge_list['venue']['paper'][relation][venue_id].keys())
        venue_paper.append([venue_id, graph.node_feature['venue'].loc[venue_id, 'name'], paper_num])
sort_array = sorted(venue_paper, key=lambda x: x[2], reverse=True)[:sample_field_num]
# print(sort_array)
# exit()
# 得到论文数最多的几个领域
sampled_venue = set([x[0] for x in sort_array])

author_venue = defaultdict(lambda: defaultdict(int))
# sampled_venue = set()
for relation in paper_relation:
    author_ids = graph.edge_list['author']['paper']['rev_'+relation].keys()
    for author_id in author_ids:
        if author_id not in seeds:
            continue
        paper_ids = graph.edge_list['author']['paper']['rev_'+relation][author_id].keys()
        for paper_id in paper_ids:
            for relation2 in P_V_relation:
                if paper_id in graph.edge_list['paper']['venue']['rev_'+relation2].keys():
                    venue_ids = graph.edge_list['paper']['venue']['rev_'+relation2][paper_id].keys()
                    venue_ids = set(venue_ids) & sampled_venue
                    # sampled_venue |= set(venue_ids)
                    # author_venue[author_id] |= set(venue_ids)
                    for venue_id in venue_ids:
                        time_id = graph.edge_list['paper']['venue']['rev_'+relation2][paper_id][venue_id]
                        if time_id > author_venue[author_id][venue_id]:
                            author_venue[author_id][venue_id] = time_id
                            author_node = {'id': author_id, 'type': 'author'}
                            venue_node = {'id': venue_id, 'type': 'venue', 'attr': relation2[3:]}
                            KG.add_edge(author_node, venue_node, relation_type='contribute', directed=False, time=time_id)
maxlen = 0
for value in author_venue.values():
    if len(value) > maxlen:
        maxlen = len(value)
# print('author-venue:', author_venue)
print('author-venue')
# print(sampled_venue)
# print(author_venue)
# exit()
# print(maxlen)

# author-paper-field
# 根据论文数选取领域集
field_paper = []
field_ids = list(graph.edge_list['field']['paper']['PF_in_L1'].keys())
for field_id in field_ids:
    paper_num = len(graph.edge_list['field']['paper']['PF_in_L1'][field_id].keys())
    # print(graph.node_feature['field'].loc[field_id, 'name'], ':', paper_num)
    field_paper.append([field_id, graph.node_feature['field'].loc[field_id, 'name'], paper_num])
sort_array = sorted(field_paper, key=lambda x: x[2], reverse=True)[:sample_field_num]
# 得到论文数最多的几个领域
sampled_field = set([x[0] for x in sort_array])
# print(sort_array)
# print(sampled_field)

# print(graph.node_feature['field'].loc[field_id, 'name'])
# exit()
# 这里只爬取L1领域，没有将L1-L5整合到L0上，因为不确定一篇论文链接L5领域的同时会不会链接对应的L0领域
# 有可能出现一个作者多个标签的情况，这里保存每个作者对应标签出现的次数
# author_field = defaultdict(lambda :defaultdict(int))
# 只保存每个作者属于的标签
author_field = defaultdict(lambda : defaultdict(int))
# sampled_field = set()
for relation in paper_relation:
    author_ids = graph.edge_list['author']['paper']['rev_'+relation].keys()
    for author_id in author_ids:
        if author_id not in seeds:
            continue
        paper_ids = graph.edge_list['author']['paper']['rev_'+relation][author_id].keys()
        for paper_id in paper_ids:
            for relation2 in P_F_relation:
                if paper_id in graph.edge_list['paper']['field']['rev_'+relation2].keys():
                    field_ids = graph.edge_list['paper']['field']['rev_'+relation2][paper_id].keys()
                    field_ids = set(field_ids) & sampled_field
                    # sampled_field |= set(field_ids)
                    # author_field[author_id] |= set(field_ids)
                    for field_id in field_ids:
                        # author_field[author_id][field_id] += 1
                        time_id = graph.edge_list['paper']['field']['rev_'+relation2][paper_id][field_id]
                        # author_field[author_id][field_id]默认是0，所以初始的时候time_id一定大于，会执行赋值操作
                        if time_id > author_field[author_id][field_id]:
                            author_field[author_id][field_id] = time_id
                            author_node = {'id': author_id, 'type': 'author'}
                            field_node = {'id': field_id, 'type': 'field', 'attr': relation2[-2:]}
                            KG.add_edge(author_node, field_node, relation_type='study', directed=False,
                                        time=time_id)
maxlen = 0
for value in author_field.values():
    if len(value) > maxlen:
        maxlen = len(value)
# print('author-field:', author_field)
print('author-field')
# print(maxlen)

# author-affiliation
author_affiliation = defaultdict(lambda : defaultdict(int))
sampled_affiliation = set()
author_ids = graph.edge_list['author']['affiliation']['rev_in'].keys()
for author_id in author_ids:
    # 没有考虑一个人改变组织的情况
    if author_id not in seeds:
        continue
     # graph.edge_list['author']['affiliation']['rev_in'][author_id].keys()
    affiliation_ids = graph.edge_list['author']['affiliation']['rev_in'][author_id].keys()
    # author_affiliation[author_id] |= set(affiliation_ids)
    sampled_affiliation |= set(affiliation_ids)
    for affiliation_id in affiliation_ids:
        # 这里的time_id全都是None
        # time_id = graph.edge_list['author']['affiliation']['rev_in'][author_id][affiliation_id]
        time_id = author_time[author_id]
        author_affiliation[author_id][affiliation_id] = time_id
        author_node = {'id': author_id, 'type': 'author'}
        affiliation_node = {'id': affiliation_id, 'type': 'affiliation'}
        KG.add_edge(author_node, affiliation_node, relation_type='in', directed=False,
                    time=time_id)

maxlen = 0
for value in author_affiliation.values():
    if len(value) > maxlen:
        maxlen = len(value)
# print('author-affiliation:', author_affiliation)
print('author-affiliation')
# print(SN_final.edge_list)
# print(KG.edge_list)
# print(maxlen)

save()
