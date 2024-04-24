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
    def __init__(self):
        super().__init__()
        self.heuristic_model = Net()
        self.predict_model = Net2()
        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv) # đưa về lấy mâu trên phân phối đều trong khoảng -stdv, stdv

    def forward(self, pyg):
        heuristic_measure = self.heuristic_model(pyg)
        log, topk = self.predict_model(pyg, heuristic_measure)
        return heuristic_measure, log, topk