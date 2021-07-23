import dgl
import torch
plays_g = dgl.bipartite([(0, 0), (1, 0), (1, 2), (2, 1)], 'user', 'plays', 'game')
follows_g = dgl.graph([(0, 1), (1, 2), (1, 2)], 'user', 'follows')
g = dgl.hetero_from_relations([plays_g, follows_g])
print(g)
g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])
sub_g = g.edge_subgraph({('user', 'follows', 'user'): [1, 2],
                        ('user', 'plays', 'game'): [2]},
                        preserve_nodes=True)
print(sub_g)