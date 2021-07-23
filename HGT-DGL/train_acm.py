import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import os
data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
data_file_path = 'ACM.mat'

# urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
# 0.5及以后的版本用不了现在的数据读取方法，且高版本不能直接输出图G
# https://blog.csdn.net/ShakalakaPHD/article/details/114526374

def get_graph(DATA = 'OAG'):
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
G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'],
        ('author', 'writing', 'paper') : data['PvsA'].transpose(),
        ('paper', 'citing', 'paper') : data['PvsP'],
        ('paper', 'cited', 'paper') : data['PvsP'].transpose(),
        ('paper', 'is-about', 'subject') : data['PvsL'],
        ('subject', 'has', 'paper') : data['PvsL'].transpose(),
    })
# G = get_graph()
print(G)
# exit()

# 返回稀疏矩阵的csr_matrix形式
pvc = data['PvsC'].tocsr()
# 返回稀疏矩阵的coo_matrix形式
p_selected = pvc.tocoo()

# generate labels
# 这是一个节点分类任务
labels = pvc.indices
labels = torch.tensor(labels).long()
# print(labels.max().item())
# exit()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

device = torch.device("cuda:0")
G.node_dict = {}
G.edge_dict = {}
# 给每个类型加上id，从0开始
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype] 
    
#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad = False).to(device)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb
    

model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item()+1, n_layers=2, n_heads=4, use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3, pct_start=0.05)

best_val_acc = 0
best_test_acc = 0
train_step = 0
for epoch in range(100):
    # 这里是在整张图上做训练
    model.train()
    logits = model(G, 'paper')
    # print(logits.shape)
    # exit()
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))

    pred = logits.argmax(1).cpu()
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc   = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc  = (pred[test_idx] == labels[test_idx]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    
    if epoch % 5 == 0:
        print('LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            optimizer.param_groups[0]['lr'], 
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))
