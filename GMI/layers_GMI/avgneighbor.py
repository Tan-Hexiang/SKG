import sys
sys.path.append('../utils')
import torch
import torch.nn as nn
# from utils import process

# Applies mean-pooling on neighbors
class AvgNeighbor(nn.Module):
    def __init__(self):
        super(AvgNeighbor, self).__init__()

    def forward(self, seq, adj_ori):
        # 稀疏矩阵转成torch类型的矩阵
        # adj_ori = process.sparse_mx_to_torch_sparse_tensor(adj_ori)
        # if torch.cuda.is_available():
        #     adj_ori = adj_ori.cuda()
        # torch.spmm相当于矩阵乘法
        # 这里相当于得到节点的邻居表示的和
        return torch.unsqueeze(torch.spmm(adj_ori, torch.squeeze(seq, 0)), 0)
