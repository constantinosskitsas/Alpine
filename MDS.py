import sys

sys.path.append("../")
import torch
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from joint_mds import JointMDS
from scipy.sparse import csr_matrix
import argparse
import pickle
import warnings
import networkx as nx
warnings.filterwarnings("ignore")
from numpy import linalg as LA
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
def calculate_node_correctness(pairs, num_correspondence):
    node_correctness = 0
    for pair in pairs:
        if pair[0] == pair[1]:
            node_correctness += 1
    node_correctness /= num_correspondence
    return node_correctness


def get_pairs_name(trans, idx2node_s, idx2node_t, weight_t=None):
    pairs_name = []

    target_idx = list(range(trans.shape[1]))
    for s in range(trans.shape[0]):
        if weight_t is not None:
            row = trans[s, :] / weight_t  # [:, 0]
        else:
            row = trans[s, :]
        idx = np.argsort(row)[::-1]
        for n in range(idx.shape[0]):
            if idx[n] in target_idx:
                t = idx[n]
                pairs_name.append([idx2node_s[s], idx2node_t[t]])
                target_idx.remove(t)
                break
    return pairs_name


def evaluate(P, idx2node_s, idx2node_t, weight_t=None):
    pairs = get_pairs_name(P, idx2node_s, idx2node_t, weight_t)
    # print(pairs)
    acc = calculate_node_correctness(pairs, len(P))
    return acc


def normalize_adj(adj):
    degree = np.asarray(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    # print("here")
    # print(d_inv_sqrt.shape)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # .tocoo()


def get_gt(idx2node_s, idx2node_t, size):
    node2idx_s = {}
    node2idx_t = {}
    for k in idx2node_s:
        node2idx_s[idx2node_s[k]] = k
        node2idx_t[idx2node_t[k]] = k
    P_true = np.zeros((size, size), dtype=bool)
    for k in node2idx_s:
        P_true[node2idx_s[k], node2idx_t[k]] = True
    return P_true


def compute_shortest_path(adj):
    adj.data = 1.0 / (1.0 + adj.data)
    #adj=1.0/(1.0+adj)
    # adj.data = 1. - adj.data
    adj = dijkstra(csgraph=adj, directed=False, return_predecessors=False)
    adj /= adj.mean()
    return adj


def get_quadratic_inverse_weight(shortest_path):
    w = 1.0 / shortest_path**4
    w[np.isinf(w)] = 0.0
    w /= w.sum()
    return w


def my_eval(P, P_true):
    return P[P_true].sum()


def MDSGA(Gq,Gt):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    nmin= min(n1,n2)  
    

    A = nx.to_numpy_array(Gq)
    B = nx.to_numpy_array(Gt)
    A = csr_matrix(A)
    B = csr_matrix(B)
    adj_s_normalized = normalize_adj(A)
    adj_t_normalized = normalize_adj(B)

    print("adj_s_normalized")
    adj_s_normalized = compute_shortest_path(adj_s_normalized)
    print("adj_s_normalized-paths")
    adj_t_normalized = compute_shortest_path(adj_t_normalized)
    print("adj_t_normalized-paths")
    w1 = get_quadratic_inverse_weight(adj_s_normalized)
    w2 = get_quadratic_inverse_weight(adj_t_normalized)
    print("w1-w2")
    w1 = torch.from_numpy(w1)
    w2 = torch.from_numpy(w2)
    torch.manual_seed(1)
    JMDS = JointMDS(
        n_components=2,
        alpha=1.0,
        #alpha=0.1,
        max_iter=500,
        eps=0.01,
        #eps=1,
        tol=1e-5,
        min_eps=0.001,
        eps_annealing=True,
        alpha_annealing=True,
        gw_init=True,
        return_stress=False
    )
    Z1, Z2, P = JMDS.fit_transform(
        torch.from_numpy(adj_s_normalized),
        torch.from_numpy(adj_t_normalized),
        w1=w1,
        w2=w2#,
    )
    cost_matrix = P.numpy()
    P2,_ = convertToPermHungarian2(cost_matrix, n1, n2)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian2(cost_matrix, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm