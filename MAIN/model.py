import dgl
import math
import torch
import torch.nn as nn
from model_HGT import *
from model_GAE import *
import torch.nn.functional as F
import dgl.function as fn
from layers import AvgNeighbor, Discriminator
import utils


class Model_HGT_GAE(nn.Module):
    def __init__(self, model_KG, model_SN):#KG_parameters, SN_parameters):
        super(Model_HGT_GAE, self).__init__()
        self.model_KG = model_KG
        self.model_SN = model_SN
        # G, in_features, hidden_features, out_features, n_layers, n_heads, use_norm = KG_parameters
        # in_dim, hidden1_dim, hidden2_dim = SN_parameters
        # self.model_KG = HGT_PF(G, in_features, hidden_features, out_features, n_layers, n_heads, use_norm)
        # self.model_SN = GAEModel(in_dim, hidden1_dim, hidden2_dim)

    def forward(self, KG, nega_KG, etype, SN, features):
        pred_pos, pred_neg = self.model_KG(KG, nega_KG, etype)
        adj_rec, feats = self.model_SN(SN, features)
        return pred_pos, pred_neg, adj_rec, feats

class Model_HGT_GAE_GMI(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_HGT_GAE_GMI, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, adj_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(KG, nega_KG, etype)
        # KG部分的对比学习
        # 对比学习是否只比较同类节点，这里将知识图谱转成同质图，即不考虑节点类型和边的类型。节点数量太多，邻接矩阵可能会炸
        # 保存节点的隐状态和线性变化向量
        HKG = dgl.to_homogeneous(KG, ndata=['h', 'w', 'feature'])
        # print(HKG)
        adj_KG = HKG.adjacency_matrix().to_dense().to(device)
        features_KG = HKG.ndata['feature']
        h_w_KG = HKG.ndata['w']
        embed_KG = HKG.ndata['h']
        h_neighbour_KG = self.prelu(self.avg_neighbor(h_w_KG, adj_KG))
        h_neighbour_KG = torch.squeeze(h_neighbour_KG, 0)
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_KG = self.disc1(embed_KG, features_KG, utils.process.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, utils.process.negative_sampling(adj_KG, neg_num))


        adj_rec, embed_SN, h_w_SN = self.model_SN(SN, features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, utils.process.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, utils.process.negative_sampling(adj_SN, neg_num))
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, adj_rec, embed_SN, res_mi_SN, res_local_SN