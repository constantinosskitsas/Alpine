
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
from grampa import Grampa
from Grad.grad import gradMain
from mcmc.mc import mcAlign
foldernames = [ 'arenas','netscience', 'multimanga', 'highschool', 'voles','facebook','dblp']
n_G = [ 1133,379, 1004, 327, 712,1034,9916,9872]
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(5)
foldernames = [ 'netscience']
n_G = [379]
iters =1
percs = [(i+1)/10 for i in range(0,10)]
#percs =[0.5]
perc=0.5
G = read_real_graph(n =327, name_ = f'./raw_data/highschool.txt')
#n=10240
#G = read_real_graph(n, name_ = f'./raw_data/random/subgraph_DG_{n}.txt')
#G = read_real_graph(n =379, name_ = f'./raw_data/arenas.txt')
#G_Q = read_real_graph(n =712, name_ = f'./raw_data/voles.txt')
accuracy=0
forb_norm1=0
for i in range(1):
    file_subgraph = f'./raw_data/random/subgraph_QG_{n}.txt'
    file_nodes = f'./raw_data/random/nodes_QG_{n}.txt'
    #file_subgraph = f'./data3_/arenas/50/5/subgraph.txt'
    #file_nodes = f'./data3_/arenas/50/5/nodes.txt'
    Q_real = read_list(file_nodes)
    n_Q = int(perc*G.number_of_nodes())              
    G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
    #G_Q = read_real_graph(n =712, name_ = f'./raw_data/voles.txt')
    start = time.time()
    _, list_of_nodes, forb_norm = Alpine(G_Q.copy(), G.copy(),mu=1,weight=2)
    print(forb_norm)
    count=0

    tempAcc= np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
    print(tempAcc)
    end = time.time()
    time_diff = end - start
    print(time_diff)
    sys.exit()
   
    
    
    tempAcc= np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
    tempAcc1= np.sum(np.array(Q_real)==np.array(list_of_nodes1))/len(Q_real)
    sorted_pairs = sorted(zip(list_of_nodes,list_of_nodes1)) 
    list1_sorted, list2_sorted = zip(*sorted_pairs)
    list1_sorted = list(list1_sorted)
    list2_sorted = list(list2_sorted)
    print(len(list1_sorted))
    print(len(list2_sorted))
    print(list1_sorted)
    print(list2_sorted)
    print(np.sum(np.array(Q_real)==np.array(list1_sorted))/len(Q_real))
    print(np.sum(np.array(Q_real)==np.array(list2_sorted))/len(Q_real))
    sys.exit() 
    accuracy =accuracy+ tempAcc
    forb_norm1=forb_norm+forb_norm1
    print(tempAcc)
    print(tempAcc1)
    

#print("Accuracy: ",accuracy/10)
#print("FN: ",forb_norm1/10)