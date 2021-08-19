import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser('SKG')
    parser.add_argument('--dataset', type=str, default='OAG',
                        choices=['OAG', 'WDT'],
                        help='Initial learning rate.')
    parser.add_argument('--KG_model', type=str, default='HGT',
                        choices=['RGCN', 'HGT'],
                        help='The model for KG embedding.')
    parser.add_argument('--SN_model', type=str, default='GAT',
                        choices=['GCN', 'GAT', 'GAE'],
                        help='The model for SN embedding.')
    # 用cos效果好的比较明显，但是实体对齐没有啥变化
    parser.add_argument('--align_dist', type=str, default='L1',
                        help='The type of align nodes distance.',
                        choices=['L1', 'L2', 'cos'])
    parser.add_argument('--task', type=str, default='SN_NC',
                        # NC:Node Classification, LP:Link Prediction
                        choices=['KG_NC', 'KG_LP', 'SN_NC', 'SN_LP', 'Align'],
                        help='The main task.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--in_dim', type=int, default=768,
                        help='Dim of input embedding vector.')
    # 隐藏层向量维度
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Number of units in hidden layer.')
    # 输出向量维度，LP中都使用hidden_dim作为输出维度，NC中使用n_classes，很少使用这个维度
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Dim of output embedding vector.')
    # 负采样的比例
    parser.add_argument('--neg_num', type=int, default=1,
                        help='Number of negtive sampling of each node.')
    # margin-loss的margin
    parser.add_argument('--margin', type=int, default=1,
                        help='The margin of alignment for entities.')
    # 实体对齐参数
    parser.add_argument('--align_sample', type=int, default=10,
                        help='the sample number in align evaluation, head and tail for twice')
    # hits@X的X
    parser.add_argument('--hits_num', type=int, default=10,
                        help='hits@num')
    # 各个损失函数的权重
    parser.add_argument('--w_KG', type=float, default=1,
                        help='weight for KG downstream task')
    parser.add_argument('--w_SN', type=float, default=0.5,
                        help='weight for SN downstream task')
    parser.add_argument('--w_KG_MI', type=float, default=0.7,
                        help='weight for KG mutual information loss')
    parser.add_argument('--w_SN_MI', type=float, default=0.5,
                        help='weight for SN mutual information loss')
    parser.add_argument('--w_align', type=float, default=0.8,
                        help='weight for node alignment between two graphs')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='parameter for I(h_i; x_i) (default: 0.8)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU id to use, <0 means cpu.')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train set ratio.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Valid set ratio.')
    # 比当前结果提升tolerance才更新最优结果
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='toleratd margainal improvement for early stopper')
    # max_round轮中没有比现在更大的，就early stop
    parser.add_argument('--max_round', type=float, default=20,
                        help='the max round for early stopping')
    # min_epoch后才能开始early stop
    parser.add_argument('--min_epoch', type=float, default=250,
                        help='the min epoch before early stopping')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    # GCN聚合的层数
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--use_self_loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    # GAT需要的参数
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    # 不加这个最后的结果维度会是正常的num-heads倍
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv