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
from AMO import Net3

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


class ACO():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 time_window, # (n, 3)
                 pyg,
                 k,
                 model,
                 log,
                 topk,
                 capacity,
                 drop_start_node = False,
                 n_ants=50,
                 decay=0.9,
                 alpha=2,
                 beta=0.5,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device=device,
                 adaptive=False,
                 prob = 0.3
                 ):
        self.drop_start_node = drop_start_node
        self.prob = prob,
        self.log = log,
        self.topk = topk,
        self.k = k
        self.time_window = time_window
        self.model = model
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demand = demand
        self.pyg = pyg
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or adaptive
        self.min_max = min_max
        self.adaptive = adaptive

        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None

        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances, device = device)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        if self.adaptive:
            self.elite_pool = []

        self.heuristic = 1 / distances if heuristic is None else heuristic # TODO

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    def sample(self): #training => ok
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return paths, costs, log_probs

    def final_sol(self, paths, costs, log_probs):
        pass


    @torch.no_grad()
    def run(self, n_iterations): # ok
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)

            if self.adaptive: # Local_search
                self.improvement_phase(paths, costs)

            improved = False
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
                if self.adaptive:
                    self.intensification_phase(paths, costs, best_idx)
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max / self.pheromone.max()
                    self.max = max
                improved = True

            if not self.adaptive or improved:
                self.update_pheronome(paths, costs)
                if self.adaptive:
                    self.elite_pool.insert(0, (self.shortest_path, self.lowest_cost))
                    if len(self.elite_pool) > 5:  # pool_size = 5
                        del self.elite_pool[5:]
            else:
                self.diversification_phase()

        return self.lowest_cost

    @torch.no_grad()
    def update_pheronome(self, paths, costs): # ok
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost

        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max

        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_path_costs(self, paths): # training => ok
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self, require_prob=False): # training => ok
        actions = torch.zeros((self.n_ants * self.k,), dtype=torch.long, device=self.device)

        visit_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants * self.k,), device=self.device)
        used_time = torch.zeros(size=(self.n_ants * self.k,), device=self.device)

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        used_time, time_mask = self.update_time_mask(actions, actions, used_time)


        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants

        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        done = self.check_done(visit_mask, actions)
        # first_start
        if self.drop_start_node == False:
            for _ in range(1):
                pre_node = copy.deepcopy(actions)
                actions, log_probs = self.topk_start_move(require_prob)
                paths_list.append(actions)
                if require_prob:
                    log_probs_list.append(log_probs)
                    visit_mask = visit_mask.clone()
                visit_mask = self.update_visit_mask(visit_mask, actions)
                used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
                used_time, time_mask = self.update_time_mask(actions, pre_node, used_time)

                done = self.check_done(visit_mask, actions)

        while not done:
            pre_node = copy.deepcopy(actions)
            actions, log_probs = self.pick_move(actions, visit_mask, capacity_mask, time_mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            used_time, time_mask = self.update_time_mask(actions,pre_node, used_time)


            done = self.check_done(visit_mask, actions)

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)


    def topk_start_move(self, require_prob): # training => ok
        actions = self.topk[0].repeat(self.n_ants) # (n_ants * k, )
        log_probs = self.log[0].repeat(self.n_ants) if require_prob else None # (n_ants * k, )
        return actions, log_probs

    def pick_move(self, prev, visit_mask, capacity_mask, time_mask, require_prob): # traing => ok
        pheromone = self.pheromone[prev].to(device) # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev].to(device) # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask *  capacity_mask * time_mask) # shape: (n_ants, p_size)

        dist_copy = torch.where(dist == 0, -1e20, dist).to(device)
        dist_copy += 1000
        dist_1 = Categorical(logits = dist_copy)
        r = torch.rand(1)
        if r < 0.3:
            actions = dist_1.sample() # shape: (n_ants,)

        else:
            actions = torch.argmax(dist_copy, dim = 1)

        log_probs = dist_1.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs

    def update_visit_mask(self, visit_mask, actions): # training => Ok
        visit_mask[torch.arange(self.n_ants * self.k, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask

    def update_time_mask(self, cur_nodes, pre_nodes, used_time): # training => ok
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_time: shape (n_ants, )
            time_mask: shape (n_ants, p_size)
        Returns:
            ant_time: updated capacity
            time_mask: updated mask
        '''
        time_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        # update time
        used_time = used_time + self.distances[pre_nodes,cur_nodes]
        used_time[cur_nodes==0] = 0
        start = self.time_window[cur_nodes, 0]
        used_time = torch.where(used_time < start, start, used_time)
        used_time = used_time + self.time_window[cur_nodes, [2] * self.n_ants * self.k]

        # update time mask
        time = self.distances[cur_nodes.expand([self.problem_size,-1]).T.flatten(),  torch.arange(self.problem_size).repeat(1,self.n_ants * self.k)].view(self.n_ants * self.k, self.problem_size).to(device)
        # (self.n_ants * self.k, self.problem_size)
        time = used_time.view(-1,1).expand(-1, self.problem_size) + time
        finish = self.time_window[:, 1].expand(self.n_ants * self.k, -1)
        time_mask[time > finish] = 0
        return used_time, time_mask


    def update_capacity_mask(self, cur_nodes, used_capacity): # traing => Ok
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        self.remain = remaining_capacity
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size).to(device) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants * self.k, 1).to(device) # (n_ants, p_size)
        self.used_cap = used_capacity
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0

        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions): # training => ok
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()



