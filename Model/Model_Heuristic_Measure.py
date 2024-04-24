import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn
from Gen_CVRPTW_data import *
import math 

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=6, feats_cvrp=1, feats_tw = 3,  units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        # Parametet for CVRP
        self.depth = int(depth)
        self.feats_cvrp = int(feats_cvrp)
        self.units = int(units)
        self.act_fn_cvrp = getattr(F, act_fn)
        self.agg_fn_cvrp = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0_cvrp = nn.Linear(self.feats_cvrp, self.units)
        self.v_lins1_cvrp = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2_cvrp = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3_cvrp = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4_cvrp = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns_cvrp = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0_cvrp = nn.Linear(1, self.units)
        self.e_lins0_cvrp = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns_cvrp = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

        # Parametet for TW
        self.depth = int(depth)
        self.feats_tw  = int(feats_tw)
        self.units  = int(units)
        self.act_fn_tw  = getattr(F, act_fn)
        self.agg_fn_tw  = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0_tw  = nn.Linear(self.feats_tw, self.units)
        self.v_lins1_tw  = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2_tw  = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3_tw  = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4_tw  = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns_tw  = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0_tw  = nn.Linear(1, self.units)
        self.e_lins0_tw  = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns_tw  = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv) # đưa về lấy mâu trên phân phối đều trong khoảng -stdv, stdv



    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x,y, edge_index, edge_attr):
        # For CVRP
        x = x
        w_cvrp = deepcopy(edge_attr)
        x = self.v_lin0_cvrp(x)
        x = self.act_fn_cvrp(x)
        w_cvrp = self.e_lin0_cvrp(w_cvrp)
        w_cvrp = self.act_fn_cvrp(w_cvrp)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1_cvrp[i](x0)
            x2 = self.v_lins2_cvrp[i](x0)
            x3 = self.v_lins3_cvrp[i](x0)
            x4 = self.v_lins4_cvrp[i](x0)
            w_cvrp0 = w_cvrp
            w_cvrp1 = self.e_lins0_cvrp[i](w_cvrp0)
            w_cvrp2 = torch.sigmoid(w_cvrp0)
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            x = x0 + self.act_fn_cvrp(self.v_bns_cvrp[i](x1 + self.agg_fn_cvrp(w_cvrp2 * x2[edge_index[1]], edge_index[0].to(device))))
            w_cvrp = w_cvrp0 + self.act_fn_cvrp(self.e_bns_cvrp[i](w_cvrp1 + x3[edge_index[0]] + x4[edge_index[1]]))


        # For tw
        y = y
        w_tw = deepcopy(edge_attr)
        y = self.v_lin0_tw(y)
        y = self.act_fn_tw(y)
        w_tw = self.e_lin0_tw(w_tw)
        w_tw = self.act_fn_tw(w_tw)
        for i in range(self.depth):
            y0 = y
            y1 = self.v_lins1_tw[i](y0)
            y2 = self.v_lins2_tw[i](y0)
            y3 = self.v_lins3_tw[i](y0)
            y4 = self.v_lins4_tw[i](y0)
            w_tw0 = w_tw
            w_tw1 = self.e_lins0_tw[i](w_tw0)
            w_tw2 = torch.sigmoid(w_tw0)
            y = y0 + self.act_fn_tw(self.v_bns_tw[i](y1 + self.agg_fn_tw(w_tw2 * y2[edge_index[1]], edge_index[0].to(device))))
            w_tw = w_tw0 + self.act_fn_tw(self.e_bns_tw[i](w_tw1 + y3[edge_index[0]] + y4[edge_index[1]]))
        return (w_cvrp, w_tw)

# general class for MLP
class MLP(nn.Module):
    def __init__(self, units_list, act_fn):
        super().__init__()
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv) # đưa về lấy mâu trên phân phối đều trong khoảng -stdv, stdv


    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='relu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)
    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_heu = ParNet()
    def forward(self, pyg):
        x, y, edge_index, edge_attr = pyg.x, pyg.time_window, pyg.edge_index, pyg.edge_attr
        emb_cvrp, emb_tw = self.emb_net(x, y, edge_index, edge_attr)
        heu = self.par_net_heu((emb_cvrp * emb_tw))
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        '''Turn phe/heu vector into matrix with zero padding
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix
