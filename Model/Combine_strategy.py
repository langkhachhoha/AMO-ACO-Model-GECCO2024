import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn

class MultiPlication_Strategy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x * y
    