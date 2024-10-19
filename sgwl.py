# from .model.GromovWassersteinLearning import GromovWassersteinLearning
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torch
#original code https://github.com/HongtengXu/s-gwl
import numpy as np
import scipy.sparse as sps
import scipy
# import scipy.sparse as sps
from methods import DataIO, GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import torch
from numpy import linalg as LA
# methods = ['gwl', 's-gwl-3', 's-gwl-2', 's-gwl-1']
cluster_num = [2, 4, 8]
partition_level = [3, 2, 1]
def clean_matrix(matrix):
    # Convert the matrix to a numpy array if it isn't one already
    matrix = np.array(matrix, dtype=float)  # Ensure it's a float array to handle NaN and inf properly
    
    # Replace NaN values with 0
    matrix = np.nan_to_num(matrix, nan=0.0)
    
    # Find the maximum finite value in the matrix
    max_finite_value = np.max(matrix[np.isfinite(matrix)])
    
    # Replace positive and negative infinite values with the maximum finite value
    matrix[np.isposinf(matrix)] = max_finite_value
    matrix[np.isneginf(matrix)] = max_finite_value
    
    return matrix
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
def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans
def SGWLSA(Gq,Gt, mn=1, max_cpu=40,clus=2,level=3):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    nmin= min(n1,n2)
    #for i in range(n1, n):
    #    Gq.add_node(i)
    #    Gq.add_edge(i,i)
    #for i in range(n2, n):
    #    Gt.add_node(i)
        

    A = nx.to_numpy_array(Gq)
    B = nx.to_numpy_array(Gt)




    p_s, cost_s, idx2node_s = DataIO.extract_graph_info(
        Gq, weights=None)

    p_s /= np.sum(p_s)
    p_t, cost_t, idx2node_t = DataIO.extract_graph_info(
        Gt, weights=None)
    p_t /= np.sum(p_t)
    if max_cpu > 0:
        torch.set_num_threads(max_cpu)
    ot_dict= {
        'loss_type': 'L2',  # the key hyperparameters of GW distance
        'ot_method': 'proximal',
        #'beta': 0.1,
        'beta': 0.2,
        'iter_bound': 1e-10,
        'inner_iteration': 2,
        'sk_bound': 1e-30,#--mine
        'node_prior': 1000,#--mine
        
        'max_iter': 4,#--mine  # iteration and error bound for calcuating barycenter
        'cost_bound': 1e-26,
        'update_p': False,  # optional updates of source distribution
        'lr': 0,
        'alpha': 1
    }
    ot_dict = {
        **ot_dict,
        'outer_iteration': n
    }
    
    
    if mn == 0:
        pairs_idx, pairs_name, pairs_confidence, trans = GwGt.direct_graph_matching(
            0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t, idx2node_s, idx2node_t, ot_dict)
    else:
        pairs_idx, pairs_name, pairs_confidence, trans = GwGt.recursive_direct_graph_matching(
            0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t,
            idx2node_s, idx2node_t, ot_dict, weights=None, predefine_barycenter=False,
            cluster_num=clus, partition_level=level, max_node_num=200
        )
    #pairs = np.array(pairs_name)[::-1].T
    cost_matrix=trans*1
    cost_matrix=clean_matrix(cost_matrix)
    P2,_ = convertToPermHungarian2(cost_matrix, n1, n2)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian2(cost_matrix, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm
    cost_matrix=trans
    P2,_ = convertToPermHungarian(cost_matrix, n, n)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian(cost_matrix, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm
