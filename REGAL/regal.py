from random import random
import networkx as nx
import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
import scipy.sparse as sps
from memory_profiler import profile

#original code from https://github.com/GemsLab/REGAL
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix

from REGAL.xnetmf import get_representations
from REGAL.config import RepMethod, Graph
from REGAL.alignments import get_embeddings,get_embeddings1, get_embedding_similarities
import scipy
from numpy import linalg as LA
import torch

def convertToPermHungarian2(M, n, m):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    #P = torch.zeros((n,m), dtype = torch.float64)
    P= np.zeros((n,m))
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n) or (col_ind[i] >= m):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def G_to_Adj1(G1, G2):
    # Get the sizes of the two graphs
    size1 = G1.shape[0]
    size2 = G2.shape[0]
    
    # Create the combined adjacency matrix
    adj = np.zeros((size1 + size2, size1 + size2), dtype=np.int8)
    
    # Place G1 in the top-left block
    adj[:size1, :size1] = G1
    
    # Place G2 in the bottom-right block
    adj[size1:, size1:] = G2
    
    return adj

def G_to_Adj(G1, G2):
    # adj1 = sps.kron([[1, 0], [0, 0]], G1)
    # adj2 = sps.kron([[0, 0], [0, 1]], G2)
    adj1 = np.array([[1, 0], [0, 0]], dtype=np.int8)
    adj1 = np.kron(adj1, G1)
    adj2 = np.array([[0, 0], [0, 1]], dtype=np.int8)
    adj2 = np.kron(adj2, G2)
    adj = adj1 + adj2
    # adj.data = adj.data.clip(0, 1)
    adj = adj.clip(0, 1)
    return adj


def Regal(Gq,Gt):
    dummy=False
    args = {
    'attributes': None,
    'attrvals': 2,
    'dimensions': 128,  # useless
    'k': 10,            # d = klogn
    'untillayer': 2,    # k
    'alpha': 0.01,      # delta
    'gammastruc': 1.0,
    'gammaattr': 1.0,
    'numtop': 10,
    'buckets': 2
    }
    # adj = G_to_Adj(Src, Tar).A
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    nmin= min(n1,n2)
    if (dummy):
        for i in range(n1, n):
            Gq.add_node(i)
            Gq.add_edge(i,i)
        for i in range(n2, n):
            Gt.add_node(i)
        

    A = nx.to_numpy_array(Gq)
    B = nx.to_numpy_array(Gt)
    if dummy:
        adg =G_to_Adj(A, B)
    else:

        adj = G_to_Adj1(A, B)

    # global REGAL_args
    # REGAL_args = parse_args()


    embed = learn_representations(adj, args)
    if(dummy):
        emb1, emb2 = get_embeddings(embed)
    else:
        emb1, emb2 = get_embeddings1(embed,n1)
    alignment_matrix, cost_matrix = get_embedding_similarities(
        emb1, emb2, num_top=10)
    cost_matrix=cost_matrix*-1
    P2,_ = convertToPermHungarian2(cost_matrix, n1, n2)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian2(cost_matrix, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm


# Should take in a file with the input graph as edgelist (REGAL_args['input)
# Should save representations to REGAL_args['output

def learn_representations(adj, REGAL_args):
    graph = Graph(adj, node_attributes=REGAL_args['attributes'])
    max_layer = REGAL_args['untillayer']
    if REGAL_args['untillayer'] == 0:
        max_layer = None
    alpha = REGAL_args['alpha']
    num_buckets = REGAL_args['buckets']  # BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer=max_layer,
                           alpha=alpha,
                           k=REGAL_args['k'],
                           num_buckets=num_buckets,
                           normalize=True,
                           gammastruc=REGAL_args['gammastruc'],
                           gammaattr=REGAL_args['gammaattr'])
    if max_layer is None:
        max_layer = 1000
    representations = get_representations(graph, rep_method)
    return representations


# pickle.dump(representations, open(REGAL_args['output, "w"))


def recovery(gt1, mb):
    nodes = len(gt1)
    count = 0
    for i in range(nodes):
        if gt1[i] == mb[i]:
            count = count + 1
    return count / nodes
