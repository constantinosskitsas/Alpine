import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler
import pandas as pd
from Grad.utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from Grad.models import *
import numpy as np
import random
import torch
from Grad.data import *
import sys
from pred import convertToPermHungarian2A
from numpy import linalg as LA
def gradMain(Gq, Gt, mu=1, niter=10, weight=1.0):
    np.random.seed(0)
    G1=Gt.copy()
    G2=Gq.copy()
    hid_dim=150
    n=G1.number_of_nodes()
    m=G2.number_of_nodes()
    for node in G1.nodes():
        if G1.degree(node) == 0:  # Check if the node has a degree of 0
            G1.add_edge(node, node)
    for node in G2.nodes():
        if G2.degree(node) == 0:  # Check if the node has a degree of 0
            G2.add_edge(node, node)
    attr1 = np.ones((len(G1.nodes),1))
        #feature_extraction1(G1)
    attr2 = np.ones((len(G2.nodes),1))

    idx1_dict = {}
    for i in range(len(G1.nodes)): idx1_dict[i] = i
    idx2_dict = {}
    for i in range(len(G2.nodes)): idx2_dict[i] = i
    alignment_dict=idx2_dict
    alignment_dict_reversed=idx2_dict
    #G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict = na_dataloader(args)
    GradAlign1 = GradAlign(G1, G2, attr1 , attr2, 2, 150,alignment_dict,alignment_dict_reversed,0,idx1_dict, idx2_dict, alpha = G2.number_of_nodes() / G1.number_of_nodes(), beta = 1)    
    seed_list1, seed_list2 = GradAlign1.run_algorithm()
    seed_list2=np.array(seed_list2)
    seed_list1=np.array(seed_list1)
    sorted_indices = np.argsort(seed_list2)

# Reorder list_of_nodes2 using the sorted indices
    list_of_nodes2_sorted = seed_list2[sorted_indices]
    list_of_nodes2_sorted=[]
    #print(len(G1.nodes))
    for i in range(len(G1.nodes)):
        list_of_nodes2_sorted.append(i)
# Reorder list_of_nodes1 with the same indices
    list_of_nodes1_sorted = seed_list1[sorted_indices]
    P2,_=convertToPermHungarian2A(list_of_nodes2_sorted,list_of_nodes1_sorted,m,n)
    A = nx.to_numpy_array(Gq)
    B = nx.to_numpy_array(Gt)
    forbnorm = LA.norm(A[:m,:m] - (P2@B@P2.T)[:m,:m], 'fro')**2
    #self.alignment_dict[seed_list1[i]] == seed_list2[i]:
    return list_of_nodes1_sorted,forbnorm


