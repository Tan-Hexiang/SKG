import numpy as np
import torch
import scipy.sparse as sp

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)



def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        # print(_type)
        # print(layer_data)
        # from sys import exit
        # exit()
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        # print(feature[_type].shape)
        # print(graph.node_feature[_type].loc[idxs, 'emb'])
        # print(graph.node_feature[_type].loc[idxs, 'citation'].shape())
        # idxs中有些id在node_feature的index中不存在，所以会返回NAN
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts

def feature_SN(layer_data, graph):
    # 新图的采样需要经过一次id转换
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        # print(_type)
        # print(layer_data)
        # from sys import exit
        # exit()
        if len(layer_data[_type]) == 0:
            continue
        # idxs  = np.array(list(layer_data[_type].keys()))
        idxs = []
        for id in layer_data[_type].keys():
            idxs.append(graph.node_backward[_type][id]['id'])
        idxs = np.array(idxs)
        # print(idxs)
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        # print(feature[_type].shape)
        # print(graph.node_feature[_type].loc[idxs, 'emb'])
        # print(graph.node_feature[_type].loc[idxs, 'citation'].shape())
        # idxs中有些id在node_feature的index中不存在，所以会返回NAN
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts