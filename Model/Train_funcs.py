import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
import copy
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from Gen_CVRPTW_data import *
from Model_Heuristic_Measure import *
from GraphAttetionEncoder import *
import math 
import numpy as np
from Model_predict_start_node import *
from ACO_training import *
import time
import matplotlib.pyplot as plt 


EPS = 1e-10
lr = 3e-4

def final_sol(paths, costs, log_probs, n_ants,  k):
    costs, indices = torch.topk(costs, k*n_ants, largest=False)
    return paths.T[indices].T.to(device), costs.to(device), log_probs.T[indices].T.to(device)


def train_instance(cfg, model, optimizer, device):
    model.train()
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

    heuristic_measure = heuristic_measure.reshape((cfg.graph_size + 1, cfg.graph_size + 1)) + EPS
    aco = ACO(distances, # (n, n)
                 demands,   # (n, )
                 time_window, # (n, 3)
                 pyg_data,
                 cfg.k_start_nodes,
                 model,
                 log,
                 topk,
                 drop_start_node = cfg.drop,
                 n_ants=cfg.n_ants,
                capacity = capacity,
                heuristic=heuristic_measure)

    paths, costs, log_probs = aco.sample()
    paths, costs, log_probs = final_sol(paths, costs, log_probs, cfg.n_ants, cfg.k_start_nodes)
    baseline = costs.mean()
    reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / (aco.n_ants*2)
    optimizer.zero_grad()
    reinforce_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def train_epoch(cfg, model, optimizer, device):
    for _ in range(cfg.steps_per_epoch):
        train_instance(cfg, model, optimizer, device)
        print("Step: ", _)




def validation_data(size = 10):
    valid_data = []
    valid_data_normalize = []
    file_list = ['C109.txt', 'C208.txt', 'R109.txt', 'R209.txt', 'RC108.txt', 'RC208.txt']
    for file_path in file_list:
        CVRPTW = gen_valid_data(file_path)
        tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
        demands = torch.cat((torch.tensor([0], device =  device), CVRPTW.demand), dim = 0)
        time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
        durations = torch.cat((torch.tensor([0], device =  device), CVRPTW.durations), dim = 0)
        time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
        capacity = CVRPTW.capacity
        distances = gen_distance_matrix(tsp_coordinates, device)
        pyg_data = gen_pyg_data(demands, time_window, durations, distances, device)
        pyg_data_normalize = gen_pyg_data_normalize(demands, time_window, durations, distances, device)
        valid_data.append((distances, demands, time_window, pyg_data, capacity))
        valid_data_normalize.append(pyg_data_normalize)
    return valid_data, valid_data_normalize

valid_data, valid_data_normalize = validation_data()

def valid_inference(net, cfg):
    result = []
    for data, data_normal in zip(valid_data, valid_data_normalize):
        heuristic_measure, log, topk = net(data_normal)
        heuristic_measure = heuristic_measure.reshape((cfg.graph_size + 1, cfg.graph_size+1)) + EPS
        distances, demands, time_window, pyg_data, capacity = data
        aco = ACO(distances, # (n, n)
                 demands,   # (n, )
                 time_window, # (n, 3)
                 pyg_data,
                 cfg.k_start_nodes,
                 net,
                 log,
                 topk,
                 drop_start_node=cfg.drop,
                 n_ants=cfg.n_ants,
                capacity=capacity,  
                heuristic=heuristic_measure)

        paths, costs, log_probs = aco.sample()
        paths, costs, log_probs = final_sol(paths, costs, log_probs, cfg.n_ants, cfg.k_start_nodes)
        costs_best, indices = torch.topk(costs, 1, largest=False)
        baseline = costs.mean()
        result.append((baseline, costs_best))
    return result


def training(cfg):
    net = Net3(k = cfg.k_start_nodes ,drop_start_node=cfg.drop).to(device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), cfg.lr)
    for epoch in range(0, cfg.epochs):
        train_epoch(cfg, net, optimizer, device )
        result = valid_inference(net, cfg)
        for i in range(len(result)):
            baseline, costs_best = result[i]
            print("Data {}: Cost_average = {}  ---  Best = {}".format(i+1, baseline, costs_best))
        print("----------------")
        print('Epoch: ', epoch)
        torch.save(net.state_dict(), cfg.checkpoint)






