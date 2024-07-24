import argparse
from train import *
from Gen_CVRPTW_data import *

def get_args():
    parser = argparse.ArgumentParser(description="AMO-ACO")
    parser.add_argument("--graph_size", type=int, default=100, help="Number of nodes in the graph")
    parser.add_argument("--data_path", type=str, default="Data/", help="the root folder of the data")
    parser.add_argument("--epochs", default=500, type=int, help="Total number of epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--n_ants", default=100, type=int, help="number of ants")
    parser.add_argument("--steps_per_epoch", default=100, type=int, help="number of steps per epoch")
    parser.add_argument("--checkpoint", type=str, default='AMO-ACO-train.pt', help="path to model checkpoint file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(device)
    args = get_args()
    training(args)
