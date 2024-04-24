import os
import sys
import pickle
import argparse
from collections import namedtuple
import numpy as np
import torch
from torch_geometric.data import Data
from Config import *

# device =  torch.device("cuda:0" if True == True else "cpu")
device = None

TW_CAPACITIES = {
        10: 250.,
        20: 500.,
        50: 750.,
        100: 1000.,
        200: 1500.,
        400: 2000.
    }




CVRPTW_SET = namedtuple("CVRPTW_SET",
                        ["depot_loc",    # Depot location
                         "node_loc",     # Node locations
                         "demand",       # demand per node
                         "capacity",     # vehicle capacity (homogeneous)
                         "depot_tw",     # depot time window (full horizon)
                         "node_tw",      # node time windows
                         "durations",    # service duration per node
                         "service_window",  # maximum of time units
                         "time_factor"])    # value to map from distances in [0, 1] to time units (transit times)




def generate_cvrptw_data(cfg, rnds=None,
                         ):
    """Generate data for CVRP-TW

    Args:
        size (int): size of dataset
        graph_size (int): size of problem instance graph (number of customers without depot)
        rnds : numpy random state
        service_window (int): maximum of time units
        service_duration (int): duration of service
        time_factor (float): value to map from distances in [0, 1] to time units (transit times)
        tw_expansion (float): expansion factor of TW w.r.t. service duration

    Returns:
        List of CVRP-TW instances wrapped in named tuples
    """
    rnds = np.random if rnds is None else rnds

    # sample locations
    dloc = rnds.uniform(size=(cfg.size, 2))  # depot location
    nloc = rnds.uniform(size=(cfg.size, cfg.graph_size, 2))  # node locations

    # TW start needs to be feasibly reachable directly from depot
    min_t = np.ceil(np.linalg.norm(dloc[:, None, :]*cfg.time_factor - nloc*cfg.time_factor, axis=-1)) + 1
    # TW end needs to be early enough to perform service and return to depot until end of service window
    max_t = np.ceil(np.linalg.norm(dloc[:, None, :]*cfg.time_factor - nloc*cfg.time_factor, axis=-1) + cfg.service_duration) + 1

    # horizon allows for the feasibility of reaching nodes / returning from nodes within the global tw (service window)
    horizon = list(zip(min_t, cfg.service_window - max_t))
    epsilon = np.maximum(np.abs(rnds.standard_normal([cfg.size, cfg.graph_size])), 1 / cfg.time_factor)

    # sample earliest start times a
    a = [rnds.randint(*h) for h in horizon]

    tw = [np.transpose(np.vstack((rt,  # a
                                  np.minimum(rt + cfg.tw_expansion * cfg.time_factor * sd, h[-1]).astype(int)  # b
                                  ))).tolist()
          for rt, sd, h in zip(a, epsilon, horizon)]

    return [CVRPTW_SET(*data) for data in zip(
        torch.tensor(dloc.tolist(), device = device) * cfg.time_factor,
        torch.tensor(nloc.tolist(), device = device) * cfg.time_factor,
        torch.tensor(np.minimum(np.maximum(np.abs(rnds.normal(loc=cfg.loc, scale=cfg.scale, size=[cfg.size, cfg.graph_size])).astype(int), 1), cfg.max).tolist(), device = device),
        torch.tensor(np.full(cfg.size, TW_CAPACITIES[cfg.graph_size]).tolist(), device = device),
        torch.tensor([[0, cfg.service_window]] * cfg.size, dtype = torch.float32, device = device),
        torch.tensor(tw, dtype = torch.float32, device = device),
        torch.tensor(np.full([cfg.size, cfg.graph_size], cfg.service_duration).tolist(), device = device),
        torch.tensor([cfg.service_window] * cfg.size, device = device),
        torch.tensor([cfg.time_factor] * cfg.size, device = device),
    )]

cfg = Data_400()
CVRPTW = generate_cvrptw_data(cfg)[0]
tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
demands = torch.cat((torch.tensor([0], device =  device), CVRPTW.demand), dim = 0)
time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
durations = torch.cat((torch.tensor([0], device =  device), CVRPTW.durations), dim = 0)
time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
service_window = CVRPTW.service_window
time_factor = CVRPTW.time_factor
capacity = CVRPTW.capacity

def gen_distance_matrix(tsp_coordinates, device):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    return distances

def gen_pyg_data(cfg, demands, time_window, durations, service_window, time_factor, distances, device):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    x = torch.where(x == 0, 1e-7, x)
    edge_attr = torch.where(edge_attr == 0, 1e-7, edge_attr)
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index, time_window = time_window, durations = durations, service_window = service_window, time_factor = time_factor)
    return pyg_data

def gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device, scale = 1000.0):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    x = torch.where(x == 0, 1e-7, x)
    edge_attr = torch.where(edge_attr == 0, 1e-7, edge_attr)
    pyg_data = Data(x=x.unsqueeze(1)/cfg.time_factor, edge_attr=edge_attr/cfg.time_factor, edge_index=edge_index, time_window = time_window/cfg.service_window, durations = durations/cfg.service_window, service_window = service_window, time_factor = time_factor)
    return pyg_data


