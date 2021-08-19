from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# from train_GAE import device
class GAE_NC(nn.Module):
    def __init__(self, SN, in_dim, hidden_dim, out_dim, device):
        super(GAE_NC, self).__init__()
        self.g = SN
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        layers = [GraphConv(self.in_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, self.out_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, self.out_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)

    def encoder(self, features):
        h = self.layers[0](self.g, features)
        h = self.layers[1](self.g, h)
        h = self.layers[2](self.g, h)
        return h

    def encoder_VGAE(self, features):
        h = self.layers[0](self.g, features)
        self.mean = self.layers[1](self.g, h)
        self.log_std = self.layers[2](self.g, h)
        gaussian_noise = torch.randn(features.size(0), self.out_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z, h

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, features):
        z, h = self.encoder_VGAE(features)
        # adj_rec = self.decoder(z)
        # 将特征进行线性变换，用来进行对比学习
        seq_fts = self.fc(features)
        # z是分类结果，h是隐藏层向量
        return z, h, seq_fts

class GAE_LP(nn.Module):
    def __init__(self, SN, in_dim, hidden_dim, out_dim, device):
        super(GAE_LP, self).__init__()
        self.g = SN
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        layers = [GraphConv(self.in_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden_dim, self.out_dim, activation=F.relu, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def encoder(self, features):
        h = self.layers[0](self.g, features)
        h = self.layers[1](self.g, h)
        h = self.layers[2](self.g, h)
        return h

    def encoder_VGAE(self, features):
        h = self.layers[0](self.g, features)
        self.mean = self.layers[1](self.g, h)
        self.log_std = self.layers[2](self.g, h)
        gaussian_noise = torch.randn(features.size(0), self.out_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, features):
        z = self.encoder_VGAE(features)
        adj_rec = self.decoder(z)
        # 将特征进行线性变换，用来进行对比学习
        seq_fts = self.fc(features)
        return adj_rec, z, seq_fts
