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
from Config import *
import matplotlib.pyplot as plt 


# device =  torch.device("cuda:0" if True == True else "cpu")
EPS = 1e-10
lr = 3e-4
device = None
cfg = Data_100() 

def final_sol(paths, costs, log_probs, n_ants,  k = 5):
    '''
        paths: (max_lenth , (n_ants * topk))
        costs: (n_ants * topk, )
        log_probs = (max_lenth - 1, (n_ants * topk))

        return:
        paths: (max_lenth, (n_ants * 2))
        costs: (n_ants * 2, )
        log_probs = (max_lenth - 1, (n_ants * 2))
    '''
    costs, indices = torch.topk(costs, k*n_ants, largest=False)
    return paths.T[indices].T.to(device), costs.to(device), log_probs.T[indices].T.to(device)


def train_instance(cfg, n, model, optimizer, n_ants, CAPACITY, device):
    model.train()
    CVRPTW = generate_cvrptw_data(cfg)[0]
    tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
    demands = torch.cat((torch.tensor([0]).to(device), CVRPTW.demand), dim = 0)
    time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
    durations = torch.cat((torch.tensor([0]).to(device), CVRPTW.durations), dim = 0)
    time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
    service_window = CVRPTW.service_window
    time_factor = CVRPTW.time_factor
    distances = gen_distance_matrix(tsp_coordinates, device = device)
    pyg_data = gen_pyg_data(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
    pyg_data_normalize = gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
    heuristic_measure, log, topk = model(pyg_data_normalize)

    heuristic_measure = heuristic_measure.reshape((n+1, n+1)) + EPS
    aco = ACO(distances, # (n, n)
                 demands,   # (n, )
                 time_window, # (n, 3)
                 pyg_data,
                 10,
                 model,
                 log,
                 topk,
              capacity = CAPACITY,
              heuristic=heuristic_measure)

    paths, costs, log_probs = aco.sample()
    paths, costs, log_probs = final_sol(paths, costs, log_probs, n_ants)
    baseline = costs.mean()
    reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / (aco.n_ants*2)
    optimizer.zero_grad()
    reinforce_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

def train_epoch(cfg, n,
                model,
                optimizer, n_ants, CAPACITY, device, steps_per_epoch
                ):
    for _ in range(steps_per_epoch):
        train_instance(cfg, n, model, optimizer, n_ants, CAPACITY, device)
        print('Done')

def validation_data(size = 10):
    valid_data = []
    valid_data_normalize = []
    for i in range(size):
        CVRPTW = generate_cvrptw_data(cfg)[0]
        tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
        demands = torch.cat((torch.tensor([0]).to(device), CVRPTW.demand), dim = 0)
        time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
        durations = torch.cat((torch.tensor([0]).to(device), CVRPTW.durations), dim = 0)
        time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
        service_window = CVRPTW.service_window
        time_factor = CVRPTW.time_factor
        distances = gen_distance_matrix(tsp_coordinates, device = device)
        pyg_data = gen_pyg_data(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
        pyg_data_normalize = gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
        valid_data.append((distances, demands, time_window, pyg_data))
        valid_data_normalize.append(pyg_data_normalize)
    return valid_data, valid_data_normalize

valid_data, valid_data_normalize = validation_data()

def valid_inference():
    result = []
    for data, data_normal in zip(valid_data, valid_data_normalize):
        heuristic_measure, log, topk = net(data_normal)
        heuristic_measure = heuristic_measure.reshape((cfg.graph_size + 1, cfg.graph_size+1)) + EPS
        distances, demands, time_window, pyg_data = data
        aco = ACO(distances, # (n, n)
                 demands,   # (n, )
                 time_window, # (n, 3)
                 pyg_data,
                 10,
                 net,
                 log,
                 topk,
                  capacity=cfg.capacity,  heuristic=heuristic_measure)

        paths, costs, log_probs = aco.sample()
        paths, costs, log_probs = final_sol(paths, costs, log_probs, n_ants = cfg.n_ants)
        costs_best, indices = torch.topk(costs, 1, largest=False)
        baseline = costs.mean()
        result.append((baseline, costs_best))
    return result


final_result = []

net = Net3().to(device)
# net.load_state_dict(torch.load(net.load_state_dict(torch.load('/kaggle/input/100-nodes/AMO_ACO_1100.pt'))))
def train(cfg, n, n_ants, steps_per_epoch, CAPACITY, epochs):
    # net = Net3().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(0, epochs):
        train_epoch(cfg, n,
                net,
                optimizer, n_ants, CAPACITY, device, steps_per_epoch
                )
        result = valid_inference()
        final_result.append([])
        for i in range(len(result)):
            baseline, costs_best = result[i]
            final_result[-1].append(result[i])
            print("Data {}: Cost_average = {}  ---  Best = {}".format(i+1, baseline, costs_best))
        print("----------------")
        print('Epoch: ', epoch)
        torch.save(net.state_dict(), os.path.join('AMO_aco_{}_train_1.pt'.format(cfg.size)))






train(cfg, cfg.graph_size, cfg.n_ants, cfg.steps_per_epoch, cfg.capacity, cfg.epochs)


