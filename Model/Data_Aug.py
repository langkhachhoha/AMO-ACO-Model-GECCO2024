import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch 

def load_data(file_path):
    xcoord=np.array([])
    ycoord=np.array([])
    demand=np.array([])
    e_time=np.array([])
    l_time=np.array([])
    s_time=np.array([])

    data = open(file_path,'r')
    lines = data.readlines()
    for i in range(len(lines)):
        if lines[i]=='NUMBER     CAPACITY\n':
            veh_num,max_cap=map(int,lines[i+1].strip().split())
        if lines[i]=='CUSTOMER\n':
            j=i+3
            while j<len(lines):
                a,b,c,d,e,f,g=map(int,lines[j].strip().split())
                xcoord=np.append(xcoord,b)
                ycoord=np.append(ycoord,c)
                demand=np.append(demand,d)
                e_time=np.append(e_time,e)
                l_time=np.append(l_time,f)
                s_time=np.append(s_time,g)
                j+=1
    cus_num=len(demand)-1

    data=[]
    for i in range(1,cus_num+2):
        new_data=[(i)]
        new_data.append((int(xcoord[i-1])))
        new_data.append((int(ycoord[i-1])))
        new_data.append((int(demand[i-1])))
        new_data.append((int(e_time[i-1])))
        new_data.append((int(l_time[i-1])))
        new_data.append((int(s_time[i-1])))
        data.append(new_data)
    

    return torch.tensor(data, dtype=torch.float32), torch.tensor(max_cap, dtype=torch.float32), veh_num


def gen_distance_matrix(tsp_coordinates, device):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    return distances


def calculate_distance_matrix(depot_loc, node_loc):
    return torch.cdist(depot_loc, node_loc, p=2)


# Phep quay 
def rotate_coordinates(depot_loc, node_loc):
    angle = np.random.uniform(0, 2*np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    depot_loc = torch.mm(depot_loc, torch.tensor(R, dtype=torch.float32))
    node_loc = torch.mm(node_loc, torch.tensor(R, dtype=torch.float32))
    return depot_loc, node_loc


# Phep lat dich
def shift_coordinates(depot_loc, node_loc):
    shift = np.random.uniform(-10, 10, size=(1, 2))
    depot_loc = torch.add(depot_loc, torch.tensor(shift, dtype=torch.float32))
    node_loc = torch.add(node_loc, torch.tensor(shift, dtype=torch.float32))
    return depot_loc, node_loc 

# Them nhieu vao toa do
def add_noise_to_coordinates(depot_loc, node_loc):
    noise_depot = np.random.uniform(-5, 5, size=(1, 2))
    noise_node = np.random.uniform(-5, 5, size=(node_loc.shape))
    depot_loc = torch.add(depot_loc, torch.tensor(noise_depot, dtype=torch.float32))
    node_loc = torch.add(node_loc, torch.tensor(noise_node, dtype=torch.float32))
    return depot_loc, node_loc

def add_noise_to_time(depot_tw, node_tw):
    noise_depot = np.random.uniform(-5, 5, size=(1, 2))
    noise_node = np.random.uniform(-5, 5, size=(node_tw.shape))
    depot_tw = torch.add(depot_tw, torch.tensor(noise_depot, dtype=torch.float32))
    node_tw = torch.add(node_tw, torch.tensor(noise_node, dtype=torch.float32))
    return depot_tw, node_tw 

# Hoan vi toa do cac node, khong hoan vi toa do depot, time window, demand 
def permute_coordinates(node_loc, demand, node_tw, durations):
    permuted_node = torch.randperm(node_loc.size(0))
    node_loc = node_loc[permuted_node]
    demand = demand[permuted_node]
    node_tw = node_tw[permuted_node]
    durations = durations[permuted_node]
    return node_loc, demand, node_tw, durations