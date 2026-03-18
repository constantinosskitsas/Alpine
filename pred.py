import numpy as np
import math
import torch
import sys
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sinkhorn import greenkhorn,sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
from numpy import linalg as LA
import networkx as nx
import time
from multiprocessing import Pool
import warnings
import ot
from memory_profiler import profile
from hung_utils import convertToPermHungarian2new,convertToPermHungarian
warnings.filterwarnings('ignore')
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)


def feature_extraction1(G,simple = True):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""
    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 2))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]
    node_features[:, 0] = degs
    node_features = np.nan_to_num(node_features)
    egonets = {n: nx.ego_graph(G, n) for n in node_list}
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]
    node_features[:, 1] = neighbor_degs
    return np.nan_to_num(node_features)

def feature_extraction(G,simple = True):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""
    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood

    if simple==False:
        neighbor_edges = [
            egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    if simple==False:
        neighbor_outgoing_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                    if not egonets[i].has_edge(*edge)
                ]
            )
            for i in node_list
        ]   

    # number of neighbors of neighbors (not in neighborhood)
    if simple==False:
        neighbors_of_neighbors = [
            len(
                set([p for m in G.neighbors(n) for p in G.neighbors(m)])
                - set(G.neighbors(n))
                - set([n])
            )
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    if (simple==False):
        node_features[:, 4] = neighbor_edges #create if statement
        node_features[:, 5] = neighbor_outgoing_edges#
        node_features[:, 6] = neighbors_of_neighbors#

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def eucledian_dist(F1, F2, n):
    #D = euclidean_distances(F1, F2)
    D = euclidean_distances(F1, F2)
    return D

def convex_init(A, B, D, mu, niter, n1):
    n = len(A)
    m = len(B)
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    P = torch.ones((n,n), dtype = torch.float64)
    P = P/n
    K=mu*D
    for i in range(niter):
        for it in range(1, 11):
            G = (torch.mm(torch.mm(A.T, A), P) - torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + torch.mm(torch.mm(P, B), B.T))/2 + mu*D + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-03)
            alpha = 2.0 / float(2.0 + it)           
            P = P + alpha * (q - P)
    P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
    P2 = torch.from_numpy(P2)
    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    return P, forbnorm,row_ind,col_ind

def align_new(Gq, Gt, mu=1, niter=10,weight=1):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    for i in range(n1, n):
        Gq.add_node(i)
        Gq.add_edge(i,i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    F1 = feature_extraction1(Gq)
    F2 = feature_extraction1(Gt)
    D = torch.zeros((n,n),dtype = torch.float64)
    D = torch.zeros((n,n),dtype = torch.float64)
    D = eucledian_dist(F1,F2,n)
    P, forbnorm,row_ind,col_ind = convex_init(A, B, D, mu, niter, n1)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm

def Alpine_pp_new(A,B, K, niter,A1,weight=1):
    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    Pi=torch.ones((m+1,n),dtype = torch.float64)
    Pi[:-1,:] *= 1/n
    Pi[-1,:] *= (n-m)/n
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m

    for i in range(10):
        for it in range(1, 11):
            deriv=(-4*I_p.T@(A-I_p@Pi@B@Pi.T@I_p.T)@I_p@Pi@B)+i*(mat_ones - 2*Pi)+K
            q=sinkhorn(ones_augm_, ones_, deriv, reg,method="sinkhorn",maxIter = 500, stopThr = 1e-5) 
            alpha = (2 / float(2 + it) )    
            Pi[:m,:n] = Pi[:m,:n] + alpha * (q[:m,:n] - Pi[:m,:n])
    Pi=Pi[:-1]
    P2,row_ind,col_ind = convertToPermHungarian(Pi, n, m)
    forbnorm = LA.norm(A - I_p[:,:m].T@P2@B@P2.T@I_p[:,:m], 'fro')**2
    return Pi, forbnorm,row_ind,col_ind

def Alpine(Gq, Gt, mu=1, niter=10, weight=2):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
        
    Gq.add_node(n1)
    Gq.add_edge(n1,n1)
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    #weight=1
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    D = torch.tensor(D, dtype = torch.float64)
    P, forbnorm,row_ind,col_ind = Alpine_pp_new(A[:n1,:n1], B, mu*D, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm    

def Fugal_pp(A, B, D, mu, niter, n1):
    n = len(A)
    m = len(B)
    P = torch.rand((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D
    for i in range(niter):
        for it in range(1, 11):
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K+ i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-5)
            alpha = 2 / float(2 + it)
            P = P + alpha * (q - P)
    P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
    P2 = torch.from_numpy(P2)
    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    return P, forbnorm,row_ind,col_ind

def Fugal(Gq, Gt, mu=1, niter=10):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
    for i in range(n1, n):
        Gq.add_node(i)
        Gq.add_edge(i,i)
    for i in range(n2, n):
       Gt.add_node(i)            
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    P, forbnorm,row_ind,col_ind = Fugal_pp(A, B, mu,D, niter,n1)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm 

def Alpine_pp_labels(A,B,feat, K, niter,A1,weight=1):
    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    Pi=torch.ones((m+1,n),dtype = torch.float64)
    feat1 = torch.tensor(feat, dtype=torch.float64, device=Pi.device)
    Pi[:-1,:] *= 1/n
    Pi[-1,:] *= (n-m)/n
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m
    gamma=1
    A0 = torch.mean(np.abs(feat1))
    dd=1
    degrees = A.sum(dim=1)
# Average degree = mean of all degrees
    avg_degree = degrees.mean()
    degrees1=B.sum(dim=1)
    avg_degree1 = degrees1.mean()
    if (avg_degree<3 or avg_degree1<3):
        dd=2
    for i in range(10):
        for it in range(1, 11):
            deriv=(-4*I_p.T@(A-I_p@Pi@B@Pi.T@I_p.T)@I_p@Pi@B)*dd+i*(mat_ones - 2*Pi)+K
            S0 = deriv.abs().mean().item()  # PyTorch version
            gamma_a = gamma * S0 / (A0+0.0001 )
            deriv = deriv + gamma_a*feat1
            q=sinkhorn(ones_augm_, ones_, deriv, reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-9) 
            alpha = (2 / float(2 + it) )    
            Pi[:m,:n] = Pi[:m,:n] + alpha * (q[:m,:n] - Pi[:m,:n])
    Pi=Pi[:-1]
    P2,row_ind,col_ind = convertToPermHungarian(Pi, n, m)
    forbnorm = LA.norm(A - I_p[:,:m].T@P2@B@P2.T@I_p[:,:m], 'fro')**2
    return Pi, forbnorm,row_ind,col_ind

def AlpineL(Gq, Gt,f1=None,f2=None, mu=1, niter=10, weight=2):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
        
    Gq.add_node(n1)
    Gq.add_edge(n1,n1)
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    feat = eucledian_dist(f1,f2,n)
    zeros_row = np.zeros((1, feat.shape[1]))
    feat=np.vstack([feat, zeros_row])
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    D = torch.tensor(D, dtype = torch.float64)
    P, forbnorm,row_ind,col_ind = Alpine_pp_labels(A[:n1,:n1], B,feat, mu*D, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm   

def Alpine_pp_new_supervised(A, B, feat, K, gtA, gtB, niter, A1, weight=1):
    """
    A: adjacency of graph A (m x m)
    B: adjacency of graph B (n x n)
    feat: feature/attribute contribution matrix (m x n)
    K: structural/regularization matrix (m+1 x n)
    gtA, gtB: ground truth anchor lists (matching nodes)
    """
    m = len(A)
    n = len(B)
    gtA = np.array(gtA, dtype=int)
    gtB = np.array(gtB, dtype=int)
    # Initialize I_p and Pi
    I_p = torch.zeros((m, m + 1), dtype=torch.float64)
    for i in range(m):
        I_p[i, i] = 1
    Pi = torch.ones((m + 1, n), dtype=torch.float64)
    Pi[:-1, :] *= 1 / n
    Pi[-1, :] *= (n - m) / n
    Pi[-1, :] = 0   

    # --- FORCE GROUND TRUTH in Pi at initialization ---
    for i, j in zip(gtA, gtB):
        Pi[i, :] = 0
        Pi[:, j] = 0
        Pi[i,j]=1
        K[i,j]=0
        #costGT[i,j]=0
    reg = 1.0
    mat_ones = torch.ones((m + 1, n), dtype=torch.float64)
    ones_ = torch.ones(n, dtype=torch.float64)
    ones_augm_ = torch.ones(m + 1, dtype=torch.float64)
    ones_augm_[-1] = n - m
    gamma = 1
    dd=1
    degrees = A.sum(dim=1)
# Average degree = mean of all degrees
    avg_degree = degrees.mean()
    degrees1=B.sum(dim=1)
    avg_degree1 = degrees1.mean()
    if (min(avg_degree,avg_degree1)<3):
        dd=2
    if (max(avg_degree,avg_degree1)>15):
        reg=10
    A0 = np.mean(np.abs(feat))
    for outer in range(10):
        for it in range(1, 11):
            deriv= (-4*I_p.T @ (A - I_p @ Pi @ B @ Pi.T @ I_p.T) @ I_p @ Pi @ B)*dd + outer * (mat_ones - 2 * Pi) + K#-SimC*5#+costGT
            S0 = deriv.abs().mean().item()  # magnitude of structural gradient
            gamma_a = gamma * S0 / (A0 + 1e-4)
            deriv = deriv + gamma_a * (feat)
            #print(np.max(gamma_a*feat))
            #deriv=deriv/10
            q=ot.sinkhorn(ones_augm_, ones_, deriv, reg, numItermax = 1000, stopThr = 1e-6)
            
            alpha = 2 / float(2 + it)
            Pi[:m, :n] = Pi[:m, :n] + alpha * (q[:m, :n] - Pi[:m, :n])
            # --- FORCE GROUND TRUTH AFTER EACH INNER ITERATION ---
            for i_gt, j_gt in zip(gtA, gtB):
                if(Pi[i_gt, j_gt] == 1):
                    continue
                Pi[i_gt, :] = 0
                Pi[:, j_gt] = 0
                Pi[i_gt, j_gt] = 1

    Pi = Pi[:-1]

    P2, row_ind, col_ind = convertToPermHungarian(Pi, n, m)
    forbnorm = LA.norm(A - I_p[:, :m].T @ P2 @ B @ P2.T @ I_p[:, :m], 'fro') ** 2

    return Pi, forbnorm, row_ind, col_ind
def Alpine_supervised(Gq, Gt,f1=None,f2=None,gtGq=None,gtGt=None, mu=1, niter=10, weight=2):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
        
    Gq.add_node(n1)
    Gq.add_edge(n1,n1)
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    feat = eucledian_dist(f1,f2,n)
    zeros_row = np.zeros((1, feat.shape[1]))
    feat=np.vstack([feat, zeros_row])
        
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    D = torch.tensor(D, dtype = torch.float64)
    
    P, forbnorm,row_ind,col_ind = Alpine_pp_new_supervised(A[:n1,:n1], B,feat,mu*D,gtGq,gtGt, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm  