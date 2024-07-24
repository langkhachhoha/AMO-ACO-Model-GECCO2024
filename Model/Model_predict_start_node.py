import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from Gen_CVRPTW_data import *
from Model_Heuristic_Measure import *
from GraphAttetionEncoder import *
import math 
import numpy as np


class Net2(nn.Module):
    def __init__(self, k_start_node = 10 ,n_heads = 4, embed_dim = 64, n_layers = 2, node_dim = 4, depth = 3 ):
        super().__init__()
        self.k = k_start_node
        self.depth = int(depth)
        self.embed_dim = int(embed_dim)
        self.attention = GraphAttentionEncoder(
            n_heads,
            embed_dim,
            n_layers,
            node_dim
        )
        self.MLP1 = nn.ModuleList([nn.Linear(int(self.embed_dim + 1), int(self.embed_dim + 1)) for i in range(self.depth)])
        self.act_fn = getattr(F, 'silu')
        self.MLP2 = nn.Linear(int(self.embed_dim + 1), int(self.embed_dim // 2))
        self.MLP3 = nn.Linear(int(self.embed_dim // 2), 1)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv) # đưa về lấy mâu trên phân phối đều trong khoảng -stdv, stdv

    def forward(self, pyg, heuristic_measure):
        """
            heuristic_measure: (n+1 * n+1, )
        """
        problem_size = pyg.x.shape[0] - 1
        # [node, 4]
        x = torch.cat( (pyg.edge_attr[1:problem_size+1], pyg.time_window[1:]), dim = 1)
        x = x.view(1, problem_size, 4) # (bathch_size, graph_size, node_dim)
        x = self.attention(x) # (batch_size, graph_size, embed_dim)
        x = x.view(problem_size, self.embed_dim)
        x = torch.cat( (x, heuristic_measure[1:problem_size+1].view(problem_size, -1)), dim = 1) # (problem_size, embed_dim + 1)
        for i in range(self.depth):
            x = self.MLP1[i](x)
            x = self.act_fn(x)
        x = self.MLP2(x)
        x = self.MLP3(x)
        x = F.log_softmax(x, dim = 0)
        x = x.view(-1)
        log, topk = torch.topk(x, self.k)
        return log, topk




        
                      



