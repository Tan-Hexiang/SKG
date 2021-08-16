import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        # 参数分别表示x1维数、x2维数和输出向量的维数
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_list, s_bias1=None, s_bias2=None):
        # sc_1表示同一个节点的向量测试结果
        # 12499*400和12499*16
        # sc_1 = torch.squeeze(self.f_k(h_pl, h_c), 2)
        # 将sc_1从N*1变成1*N
        sc_1 = torch.unsqueeze(torch.squeeze(self.f_k(h_pl, h_c), 1), 0)
        sc_1 = self.act(sc_1)
        # sc_2表示不同节点，即负样例对的向量测试结果
        sc_2_list = []
        for i in range(len(sample_list)):
            # h_mi = torch.unsqueeze(h_pl[0][sample_list[i]],0)
            h_mi = h_pl[sample_list[i]]
            # 因为双线性变换的尺寸固定了，所以负样本的数目必须要跟正样本相同
            # sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 2)
            sc_2_iter = torch.unsqueeze(torch.squeeze(self.f_k(h_mi, h_c), 1), 0)
            sc_2_list.append(sc_2_iter)
        sc_2_stack = torch.squeeze(torch.stack(sc_2_list,1),0)
        # sc_2_stack = torch.stack(sc_2_list,1)
        sc_2 = self.act(sc_2_stack)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        # 最后的sc_1和sc_2的尺寸要是1*N和neg_num*N
        return sc_1, sc_2