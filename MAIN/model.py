import dgl
import math
import torch
import torch.nn as nn
from model_HGT import *
from model_GAE import *
import torch.nn.functional as F
import dgl.function as fn


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