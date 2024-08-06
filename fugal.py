#Fugal Algorithm was provided by anonymous authors.
import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy as sci
from math import floor, log2
import math
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics.pairwise import euclidean_distances
from pred import feature_extraction,eucledian_dist,convex_init,convex_init1,euclidean_dist1
import os

def random_graph_adjacency_matrix(num_nodes, edge_prob):
    # Generate a random graph using NetworkX
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Convert the graph to an adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)

    return adj_matrix.toarray()

def main(): 
    os.environ["MKL_NUM_THREADS"] = "40"
    print("Fugal2")
    torch.set_num_threads(40)
    dtype = np.float64
    Src = random_graph_adjacency_matrix(1000,0.05)
    Tar = random_graph_adjacency_matrix(100,0.1)
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)

    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    F1= feature_extraction(Src1,True)
    F2= feature_extraction(Tar1,True)
    #D = eucledian_dist(F1, F2, n)
    D = eucledian_dist1(F1,F2,10000)
    D = torch.tensor(D, dtype = torch.float64)
    P1=convex_init1(A, B, D, 1, 15)
    return P1