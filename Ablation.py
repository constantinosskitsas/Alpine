
import numpy as np
from pred import convex_initSM, align_SM, align_new, Alpine
from help_functions import read_graph
import torch
import scipy
import networkx as nx
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
from help_functions import read_real_graph, read_list
import time
import os
from aa import generate_new_id,create_new_folder,get_max_previous_id 

foldernames = [ 'arenas','netscience', 'multimanga', 'highschool', 'voles']
n_G = [ 1133,379, 1004, 327, 712]

foldernames = [ 'voles']
n_G = [327]
iters =1
percs = [(i+1)/10 for i in range(0,10)]
#percs =[0.5]
perc=0.5
G = read_real_graph(n =327, name_ = f'./raw_data/highschool.txt')
accuracy=0
forb_norm1=0
for i in range(10):
    folder = f'./data3_/highschool_Noise15/50/{i}'
    file_subgraph = f'{folder}/subgraph.txt'
    file_nodes = f'{folder}/nodes.txt'
    Q_real = read_list(file_nodes)
    n_Q = int(perc*G.number_of_nodes())              
    G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
    _, list_of_nodes, forb_norm = Alpine(G_Q.copy(), G.copy(),mu=1,weight=1,niter=10)
    tempAcc= np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
    accuracy =accuracy+ tempAcc
    forb_norm1=forb_norm+forb_norm1
    print(tempAcc)
print("Accuracy: ",accuracy/10)
print("FN: ",forb_norm1/10)