from collections import namedtuple
import numpy as np
import torch
from torch_geometric.data import Data
import os 
from Data_Aug import *

device =  torch.device("cuda:0" if True == True else "cpu")
# device = None 

CVRPTW_SET = namedtuple("CVRPTW_SET",
                        ["depot_loc",    # Depot location
                         "node_loc",     # Node locations
                         "demand",       # demand per node
                         "capacity",     # vehicle capacity (homogeneous)
                         "depot_tw",     # depot time window (full horizon)
                         "node_tw",      # node time windows
                         "durations",    # service duration per node
                        ])   


def generate_cvrptw_data():
    file_list = os.listdir('Model/Data/')
    num_files = len(file_list)
    file_path = file_list[np.random.randint(num_files)]
    data, capacity, veh_num = load_data('Model/Data/' + file_path)
    depot_loc = data[0, 1:3][None, :]
    node_loc = data[1:, 1:3]
    demand = data[1:, 3]
    capacity = capacity 
    depot_tw = data[0, 4:6][None, :] 
    node_tw = data[1:, 4:6]
    durations = data[1:, 6]

    depot_loc, node_loc = add_noise_to_coordinates(depot_loc, node_loc)
    depot_tw, node_tw = add_noise_to_time(depot_tw, node_tw)
    
    if (np.random.rand() < 0.5):
        node_loc, demand, node_tw, durations = permute_coordinates(node_loc, demand, node_tw, durations)
    depot_loc = depot_loc.view(-1)
    depot_tw = depot_tw.view(-1)

    return CVRPTW_SET(depot_loc.to(device),
        node_loc.to(device),
        demand.to(device),
        capacity.to(device),
        depot_tw.to(device),
        node_tw.to(device),
        durations.to(device))


def gen_valid_data(file_path): 
    data, capacity, veh_num = load_data('Model/Data/' + file_path)
    depot_loc = data[0, 1:3]
    node_loc = data[1:, 1:3]
    demand = data[1:, 3]
    capacity = capacity 
    depot_tw = data[0, 4:6] 
    node_tw = data[1:, 4:6]
    durations = data[1:, 6]
    return CVRPTW_SET(depot_loc.to(device),
        node_loc.to(device),
        demand.to(device),
        capacity.to(device),
        depot_tw.to(device),
        node_tw.to(device),
        durations.to(device))





def gen_pyg_data(demands, time_window, durations, distances, device, scale = 1000.0):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    x = torch.where(x == 0, 1e-7, x)
    edge_attr = torch.where(edge_attr == 0, 1e-7, edge_attr)
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index, time_window = time_window, durations = durations)
    return pyg_data

def gen_pyg_data_normalize(demands, time_window, durations, distances, device, scale = 1000.0):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    x = torch.where(x == 0, 1e-7, x)
    edge_attr = torch.where(edge_attr == 0, 1e-7, edge_attr)
    pyg_data = Data(x=x.unsqueeze(1) / capacity, edge_attr=edge_attr/scale, edge_index=edge_index, time_window = time_window/scale, durations = durations/scale)
    return pyg_data



try:
    CVRPTW = generate_cvrptw_data()
    tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
    demands = torch.cat((torch.tensor([0], device =  device), CVRPTW.demand), dim = 0)
    time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
    durations = torch.cat((torch.tensor([0], device =  device), CVRPTW.durations), dim = 0)
    time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
    capacity = CVRPTW.capacity
    distances = gen_distance_matrix(tsp_coordinates, device)
    data = gen_pyg_data(demands, time_window, durations, distances, device)
except:
    print("Error in Gen_CVRPTW_data.py")



