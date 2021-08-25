import argparse
import sys
import time
import dill
import numpy as np

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import *#read_kgs_from_folder, read_kgs_from_OAG
from openea.models.trans import TransD
from openea.models.trans import TransE
from openea.models.trans import TransH
from openea.models.trans import TransR
from openea.models.semantic import DistMult
from openea.models.semantic import HolE
from openea.models.semantic import SimplE
from openea.models.semantic import RotatE
from openea.models.neural import ConvE
from openea.models.neural import ProjE
from openea.approaches import AlignE
from openea.approaches import BootEA
from openea.approaches import JAPE
from openea.approaches import Attr2Vec
from openea.approaches import MTransE
from openea.approaches import IPTransE
from openea.approaches import GCN_Align
from openea.approaches import AttrE
from openea.approaches import IMUSE
from openea.approaches import SEA
from openea.approaches import MultiKE
from openea.approaches import RSN4EA
from openea.approaches import GMNN
from openea.approaches import KDCoE
from openea.approaches import RDGCN
from openea.approaches import BootEA_RotatE
from openea.approaches import BootEA_TransH
from openea.approaches import AliNet
from openea.models.basic_model import BasicModel

from collections import defaultdict

# 关闭最开始的警告
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransD = TransD
    TransH = TransH
    TransR = TransR

    DistMult = DistMult
    HolE = HolE
    SimplE = SimplE
    RotatE = RotatE

    ProjE = ProjE
    ConvE = ConvE

    MTransE = MTransE
    IPTransE = IPTransE
    Attr2Vec = Attr2Vec
    JAPE = JAPE
    AlignE = AlignE
    BootEA = BootEA
    GCN_Align = GCN_Align
    GMNN = GMNN
    KDCoE = KDCoE

    AttrE = AttrE
    IMUSE = IMUSE
    SEA = SEA
    MultiKE = MultiKE
    RSN4EA = RSN4EA
    RDGCN = RDGCN
    BootEA_RotatE = BootEA_RotatE
    BootEA_TransH = BootEA_TransH
    AliNet = AliNet


def get_model(model_name):
    return getattr(ModelFamily, model_name)

def read_kgs_from_OAG(args):
    #  read_relation_triples返回三元组集合、实体集合和关系集合，其中实体和关系都是用字符串表示，三元组集合是一堆元组，(h, r, t)
    dir_path = '../dataset/OAG'

    graph = dill.load(open('../../../jiangxuhui/Project/pyHGT/dataset/oag_output/graph_ML.pk', 'rb'))
    # 这里返回KG的数据
    kg1_relation_triples = []
    author_KG = set()
    author_SN = set()
    # 这里返回每个节点的名字
    with open(os.path.join(dir_path, 'author_affiliation.txt'), 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            author_id, affiliation_id, _ = map(int, line.strip('\n').split('\t'))
            author_name = 'KG/'+graph.node_feature['author'].loc[author_id, 'name'].replace(' ', '_')
            affiliation_name = 'KG/'+graph.node_feature['affiliation'].loc[affiliation_id, 'name'].replace(' ', '_')
            author_KG.add(author_id)
            kg1_relation_triples.append((author_name, 'in', affiliation_name))
    with open(os.path.join(dir_path, 'author_field.txt'), 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            author_id, field_id, _ = map(int, line.strip('\n').split('\t'))
            author_name = 'KG/' + graph.node_feature['author'].loc[author_id, 'name'].replace(' ', '_')
            field_name = 'KG/' + graph.node_feature['field'].loc[field_id, 'name'].replace(' ', '_')
            author_KG.add(author_id)
            kg1_relation_triples.append((author_name, 'study', field_name))
    with open(os.path.join(dir_path, 'author_venue.txt'), 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            author_id, venue_id, _ = map(int, line.strip('\n').split('\t'))
            author_name = 'KG/' + graph.node_feature['author'].loc[author_id, 'name'].replace(' ', '_')
            venue_name = 'KG/' + graph.node_feature['venue'].loc[venue_id, 'name'].replace(' ', '_')
            author_KG.add(author_id)
            kg1_relation_triples.append((author_name, 'contribute', venue_name))
    # 返回SN的数据
    kg2_relation_triples = []
    with open(os.path.join(dir_path, 'author_author.txt'), 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            author_id1, author_id2, time_id = map(int, line.strip('\n').split('\t'))
            author_name1 = 'SN/' + graph.node_feature['author'].loc[author_id1, 'name'].replace(' ', '_')
            author_name2 = 'SN/' + graph.node_feature['author'].loc[author_id2, 'name'].replace(' ', '_')
            author_SN.add(author_id1)
            author_SN.add(author_id2)
            kg2_relation_triples.append((author_name1, 'co-author', author_name2))

    # 同read_relation_triples
    kg1_attribute_triples = set()
    kg2_attribute_triples = set()

    # 返回实体对齐的列表
    # 这里nodes保存的是KG中的节点id
    nodes = np.array(list(author_KG & author_SN))
    # print(len(nodes))
    np.random.shuffle(nodes)
    # 从锚节点中选取训练样本数
    # node_align_KG_train = np.random.choice(nodes, int(len(nodes)*args.train_ratio), replace=False)
    # align_ratio表示用于实体对齐的实体比例
    train_idx = int(len(nodes) * args.train_ratio * args.align_ratio)
    valid_idx = int(len(nodes) * (args.train_ratio + args.valid_ratio) * args.align_ratio)
    test_idx = int(len(nodes) * args.align_ratio)
    train_links = [('KG/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_'),
                    'SN/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_')) for node in nodes[:train_idx]]
    valid_links = [('KG/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_'),
                    'SN/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_')) for node in nodes[train_idx:valid_idx]]
    test_links = [('KG/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_'),
                    'SN/'+graph.node_feature['author'].loc[node, 'name'].replace(' ', '_')) for node in nodes[valid_idx:test_idx]]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=args.alignment_module, ordered=args.ordered)
    return kgs

def read_kgs_from_WDT(args):
    #  read_relation_triples返回三元组集合、实体集合和关系集合，其中实体和关系都是用字符串表示，三元组集合是一堆元组，(h, r, t)
    dir_path = '../dataset/WDT'

    # 这里返回KG的数据
    kg1_relation_triples = []
    wiki_KG = set()
    twitter_SN = set()
    labels = ['actor', 'dancer', 'politician', 'programmer', 'singer']
    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'country_of_citizenship.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, country = line.strip('\n').split('\t')
                wiki_KG.add(wiki)
                kg1_relation_triples.append((wiki, 'in', country))
        with open(os.path.join(dir_path, label_name, 'educated_at.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, school = line.strip('\n').split('\t')
                wiki_KG.add(wiki)
                kg1_relation_triples.append((wiki, 'educate', school))
        with open(os.path.join(dir_path, label_name, 'employer.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, affiliation = line.strip('\n').split('\t')
                wiki_KG.add(wiki)
                kg1_relation_triples.append((wiki, 'employ', affiliation))
        with open(os.path.join(dir_path, label_name, 'place_of_birth.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, birth = line.strip('\n').split('\t')
                wiki_KG.add(wiki)
                kg1_relation_triples.append((wiki, 'born', birth))
    # 返回SN的数据
    # 统计每个节点的度，去除度数较低的节点
    node_degree = defaultdict(int)
    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'follower_relation.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                # 文件中的关注关系是后一个人关注前一个人
                twitter_id2, twitter_id1 = line.strip('\n').split('\t')
                node_degree[twitter_id1] += 1
                node_degree[twitter_id2] += 1
    # 采样一个稠密子图，去掉图中度较低的节点
    node_degree = [[key, value] for key, value in node_degree.items()]
    node_degree = np.array(node_degree)
    np.random.seed(2021)
    np.random.shuffle(node_degree)
    # 保留4000个节点
    sorted_node = sorted(node_degree, key=lambda x: x[1], reverse=True)[:4000]
    sampled_nodes = [node[0] for node in sorted_node]

    kg2_relation_triples = []
    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'follower_relation.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                twitter2, twitter1 = line.strip('\n').split('\t')
                if twitter1 not in sampled_nodes or twitter2 not in sampled_nodes:
                    continue
                twitter_SN.add(twitter1)
                twitter_SN.add(twitter2)
                kg2_relation_triples.append((twitter1, 'follow', twitter2))

    # 同read_relation_triples，图中没有属性、属性值连边
    kg1_attribute_triples = set()
    kg2_attribute_triples = set()

    # 返回实体对齐的列表
    # 保存wiki和twitter的对应关系
    wiki2twitter = {}
    # nodes保存的是KG中的节点
    nodes = []
    for label_name in labels:
        with open(os.path.join(dir_path, label_name, 'twitter_and_wiki.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                wiki, twitter = line.strip('\n').split('\t')
                # 这里会出现一个实体对应多个社交帐号的情况，只保留最早出现的一个
                if wiki in wiki_KG and twitter in twitter_SN and wiki not in wiki2twitter:
                    wiki2twitter[wiki] = twitter
                    nodes.append(wiki)
    nodes = np.array(nodes)
    np.random.shuffle(nodes)
    # 从锚节点中选取训练样本数
    # align_ratio表示用于实体对齐的实体比例
    train_idx = int(len(nodes) * args.train_ratio * args.align_ratio)
    valid_idx = int(len(nodes) * (args.train_ratio + args.valid_ratio) * args.align_ratio)
    test_idx = int(len(nodes) * args.align_ratio)
    train_links = [(node, wiki2twitter[node]) for node in nodes[:train_idx]]
    valid_links = [(node, wiki2twitter[node]) for node in nodes[train_idx:valid_idx]]
    test_links = [(node, wiki2twitter[node]) for node in nodes[valid_idx:test_idx]]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=args.alignment_module, ordered=args.ordered)
    return kgs

if __name__ == '__main__':
    t = time.time()
    # python main_from_args.py ./args/alinet_args_15K.json D_W_15K_V1 721_5fold/1/
    # sys.argv[1]表示参数设置的文件地址，sys.argv[2]表示数据集的地址，sys.argv[3]表示选择的fold的地址
    model_name = 'rdgcn'
    dataset = 'OAG'
    # assert (model_name in ['gcnalign', 'rdgcn', 'alinet'])
    # args_path = './OpenEA/run/args/alinet_args_15K.json'
    # args_path = './OpenEA/run/args/gcnalign_args_15K.json'
    args_path = f'./{model_name}_args.json'
    # args = load_args(sys.argv[1])
    args = load_args(args_path)
    print(args.__dict__.items())

    # args.training_data = args.training_data + sys.argv[2] + '/'
    # args.training_data = './datasets/D_W_15K_V1/'#args.training_data + sys.argv[2] + '/'
    args.training_data = f'./datasets/{dataset}/'#args.training_data + sys.argv[2] + '/'
    args.dataset_division = '721_5fold/1/'#sys.argv[3]
    print(args.embedding_module)
    # print(args)


    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    # kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
    #                            remove_unlinked=remove_unlinked)
    if dataset == 'OAG':
        kgs = read_kgs_from_OAG(args)
    elif dataset == 'WDT':
        kgs = read_kgs_from_WDT(args)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    # 模型在测试集中做全连接排序
    # eval_metric是距离指标
    # gcn-align测试的时候只替换了尾实体，相当于OAG中每个KG实体有119个候选SN实体计算距离
    # gcn-align中se表示struct embedding，ae是attribute embedding
    # 两个知识图谱用left和right表示
    # 训练时每十轮进行一次负采样，负采样的候选集是两个图谱的所有节点，有点奇怪
    model.test()
    # model.save()
    print(args.embedding_module, '\t', dataset)
    print("Total run time = {:.3f} s.".format(time.time() - t))

    # neg_sample "truncated"
    # eval_metric 'inner' 'cosine' 'euclidean' 'manhattan'