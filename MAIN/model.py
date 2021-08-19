import dgl
import math
import torch
import torch.nn as nn
from model_HGT import *
from model_GAE import *
import torch.nn.functional as F
import dgl.function as fn
from layers_GMI import AvgNeighbor, Discriminator
from utils_GMI import process_GMI


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
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, adj_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))


        adj_rec, embed_SN, h_w_SN = self.model_SN(SN, features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_HGT_TGN_GMI(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_HGT_TGN_GMI, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        # 这一层的forward顺序是反的
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    # def forward(self, KG, nega_KG, etype, postive_graph_SN, negative_graph_SN, blocks_SN, device):
    def forward(self, KG, nega_KG, etype, neg_num, positive_graph_SN, negative_graph_SN, blocks, device):
        pred_pos_KG, pred_neg_KG = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        pred_pos_SN, pred_neg_SN, embed_SN = self.model_SN.embed(positive_graph_SN, negative_graph_SN, blocks, device)

        # SN部分的对比学习
        adj_SN = positive_graph_SN.adjacency_matrix().to_dense().to(device)
        # 这里的h_w_SN使用
        features_SN = positive_graph_SN.ndata['feature'].to(device)
        h_w_SN = self.fc(features_SN)
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        # print(embed_SN.shape, features_SN.shape)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        trans_SN = self.translation(embed_SN)

        del HKG, adj_KG, adj_SN, features_SN, h_w_SN, h_neighbour_SN, h_neighbour_KG
        return pred_pos_KG, pred_neg_KG, res_mi_KG, res_local_KG, pred_pos_SN, pred_neg_SN, trans_SN, res_mi_SN, res_local_SN

class Model_HGT_GCN_GMI(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_HGT_GCN_GMI, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, adj_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))


        adj_rec, embed_SN, h_w_SN = self.model_SN(features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_HGT_GAT_GMI(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_HGT_GAT_GMI, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, adj_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        adj_rec, embed_SN, h_w_SN = self.model_SN(features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_RGCN_GAE_GMI(nn.Module):
    # RGCN是节点分类模型
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_RGCN_GAE_GMI, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, category, SN, features_SN, adj_SN, neg_num, device):
        logits_KG = self.model_KG()[category]
        # KG部分的对比学习
        # 对比学习是否只比较同类节点，这里将知识图谱转成同质图，即不考虑节点类型和边的类型。节点数量太多，邻接矩阵可能会炸
        # 保存节点的隐状态和线性变化向量
        KG = self.model_KG.g
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        adj_rec, embed_SN, h_w_SN = self.model_SN(SN, features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return logits_KG, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_KG_NC(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):#KG_parameters, SN_parameters):
        super(Model_KG_NC, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, category, SN, features_SN, adj_SN, neg_num, device):
        logits_KG = self.model_KG()[category]
        # KG部分的对比学习
        # 对比学习是否只比较同类节点，这里将知识图谱转成同质图，即不考虑节点类型和边的类型。节点数量太多，邻接矩阵可能会炸
        # 保存节点的隐状态和线性变化向量
        KG = self.model_KG.g
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        adj_rec, embed_SN, h_w_SN = self.model_SN(features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return logits_KG, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_KG_LP(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, out_dim):
        super(Model_KG_LP, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, out_dim)
        self.disc2 = Discriminator(out_dim, out_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(out_dim, out_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, adj_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        adj_rec, embed_SN, h_w_SN = self.model_SN(features_SN)
        # SN部分的对比学习
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        # embed_SN和h_w_SN都是12499*16，h_neighbour_SN是1*12499*16
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, adj_rec, embed_SN, trans_SN, res_mi_SN, res_local_SN

class Model_SN_NC(nn.Module):
    def __init__(self, model_KG, model_SN, in_dim, hidden_dim):
        super(Model_SN_NC, self).__init__()
        # KG和SN的向量编码
        self.model_KG = model_KG
        self.model_SN = model_SN
        # SN的对比学习和互信息部分
        self.prelu = nn.PReLU()
        self.disc1 = Discriminator(in_dim, hidden_dim)
        self.disc2 = Discriminator(hidden_dim, hidden_dim)
        # 翻译层，将社交网络的向量转移到知识图谱的向量空间中
        self.translation = nn.Linear(hidden_dim, hidden_dim)
        self.avg_neighbor = AvgNeighbor()

    def forward(self, KG, nega_KG, etype, SN, features_SN, neg_num, device):
        pred_pos, pred_neg = self.model_KG(nega_KG, etype)
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
        res_mi_KG = self.disc1(embed_KG, features_KG, process_GMI.negative_sampling(adj_KG, neg_num))
        # I(h_i; x_j) node j is a neighbor
        res_local_KG = self.disc2(h_neighbour_KG, embed_KG, process_GMI.negative_sampling(adj_KG, neg_num))

        logits_SN, embed_SN, h_w_SN = self.model_SN(features_SN)
        # SN部分的对比学习
        adj_SN = SN.adjacency_matrix().to_dense().to(device)
        h_neighbour_SN = self.prelu(self.avg_neighbor(h_w_SN, adj_SN))
        h_neighbour_SN = torch.squeeze(h_neighbour_SN, 0)
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_SN = self.disc1(embed_SN, features_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # I(h_i; x_j) node j is a neighbor
        # print(embed_SN.shape, h_w_SN.shape)
        res_local_SN = self.disc2(h_neighbour_SN, embed_SN, process_GMI.negative_sampling(adj_SN, neg_num))
        # 将社交网络的节点表示翻译到知识图谱的向量空间中
        trans_SN = self.translation(embed_SN)
        return pred_pos, pred_neg, res_mi_KG, res_local_KG, logits_SN, embed_SN, trans_SN, res_mi_SN, res_local_SN

