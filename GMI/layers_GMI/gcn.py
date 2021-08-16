import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # 参数随机初始化
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        # torch.squeeze将seq_fts中维数为0的维度删去
        # torch.unsqueeze在0的位置添加一个维输为1的维度
        # 邻接矩阵和特征矩阵的乘积相当于节点的邻居和
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)

        if self.bias is not None:
            out += self.bias
        
        return self.act(out), seq_fts

