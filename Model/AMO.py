import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from Gen_CVRPTW_data import *
from Model_Heuristic_Measure import *
from GraphAttetionEncoder import *
import math 
import numpy as np
from Model_predict_start_node import *


class Net3(nn.Module):
    def __init__(self, k = 10, drop_start_node = False):
        super().__init__()
        self.heuristic_model = Net()
        self.predict_model = Net2(k_start_node=k)
        self.drop_start_node = drop_start_node
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, pyg):
        heuristic_measure = self.heuristic_model(pyg)
        if self.drop_start_node:
            log = -1
            topk = -1
        else:
            log, topk = self.predict_model(pyg, heuristic_measure)
        return heuristic_measure, log, topk 


try:
    model = Net3().to(device)
    model.eval()
    CVRPTW = generate_cvrptw_data()
    tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
    demands = torch.cat((torch.tensor([0], device =  device), CVRPTW.demand), dim = 0)
    time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
    durations = torch.cat((torch.tensor([0], device =  device), CVRPTW.durations), dim = 0)
    time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
    capacity = CVRPTW.capacity
    distances = gen_distance_matrix(tsp_coordinates, device)
    pyg_data = gen_pyg_data(demands, time_window, durations, distances, device)
    pyg_data_normalize = gen_pyg_data_normalize(demands, time_window, durations, distances, device)
    heuristic_measure, log, topk = model(pyg_data_normalize)
except:
    print("Error in train.py")

