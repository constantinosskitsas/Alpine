import numpy as np
import math
import torch
import sys
import os
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sinkhorn import greenkhorn,sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
import numpy as np
import math
import torch
from numpy import linalg as LA
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sinkhorn import sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
import warnings
import ot
from memory_profiler import profile
warnings.filterwarnings('ignore')
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)
from help_functions import read_real_graph, read_list
#
def convertToPermHungarian2(M, n, m):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    P = torch.zeros((n,m), dtype = torch.float64)
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n) or (col_ind[i] >= m):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans
    #col ind solution
##
def convertToPermHungarian2A(row_ind,col_ind,n, m):
    col_ind1=col_ind
    for i in range(n):
        if (i not in col_ind1):
                col_ind1.append(i)
    m=max(n,m)
    P = np.zeros((m,m))
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    for i in range(n):
        P[col_ind[i]][row_ind[i]] = 1
        if (row_ind[i] >= n) or (col_ind[i] >= m):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans
def convertToPermHungarianmc(row_ind,col_ind,n, m):

    m=max(n,m)
    P = np.zeros((m,m))
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    for i in range(n):
        #P[row_ind[i]][col_ind[i]] = 1
        P[col_ind[i]][row_ind[i]] = 1
        if (row_ind[i] >= m) or (col_ind[i] >= m):
            continue
        #ans.append((row_ind[i], col_ind[i]))
    return P
#used
def convertToPermHungarian2new(row_ind, col_ind, n, m):
    P = torch.zeros((n,m), dtype = torch.float64)
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n) or (col_ind[i] >= m):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

#used
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

##
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

def euclidean_dist1(F1, F2,D, maxp):
    for i in range(len(F1)):
        for j in range(len(F2)):
            if F1[i][0] > F2[j][0]:
                D[i][j] = maxp
    return D

#used
def eucledian_dist(F1, F2, n):
    #D = euclidean_distances(F1, F2)
    D = euclidean_distances(F1, F2)
    return D

#used
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

def convex_init1(A, B, D, mu, niter):
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D
    for i in range(niter):
        for it in range(1, 11):
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

#used
def convertToPermHungarian(M, n1, n2):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)
    P = np.zeros((n2, n1))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, row_ind,col_ind

def convertToPermGreedy(M, n1, n2):
    n = len(M)
    indices = torch.argsort(M.flatten())
    row_done = np.zeros(n)
    col_done = np.zeros(n)

    P = np.zeros((n, n))
    ans = []
    for i in range(n*n):
        cur_row = int(indices[n*n - 1 - i]/n)
        cur_col = int(indices[n*n - 1 - i]%n)
        if (row_done[cur_row] == 0) and (col_done[cur_col] == 0):
            P[cur_row][cur_col] = 1
            row_done[cur_row] = 1
            col_done[cur_col] = 1
            if (cur_row >= n1) or (cur_col >= n2):
                continue
            ans.append((cur_row, cur_col))
    return P, ans

#used
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
    #F1 = feature_extraction(Gq)
    #F2 = feature_extraction(Gt)
    D = torch.zeros((n,n),dtype = torch.float64)
    D = torch.zeros((n,n),dtype = torch.float64)
    D = eucledian_dist(F1,F2,n)
    P, forbnorm,row_ind,col_ind = convex_init(A, B, D, mu, niter, n1)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    #_, ans = convertToPermHungarian2(P, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm

def Alpine_pp(A, B, K, niter,A1):

    n = len(A)
    m = len(B)
    C_p = torch.zeros((m,m),dtype = torch.float64)
    for i in range(n):
        C_p[i,i] = 1
    C = C_p[:,:n]
    P = torch.rand((n,m), dtype = torch.float64)
    P = P/m
    ones = torch.ones(m, dtype = torch.float64)
    reg = 1.0
    mat_ones = torch.ones((m, m), dtype = torch.float64)

    ones_ = torch.ones(m, dtype = torch.float64)
    ones_augm_ = torch.ones(n+1, dtype = torch.float64)
    ones_augm_[-1] = m-n
    for i in range(niter):
        for it in range(1, 10):
            deriv = -2*C@A.T@C.T@P@B-2*C@A@C.T@P@B.T+2*C@(C.T@P@B@P.T@C@C.T@P@B.T+C.T@P@B.T@P.T@C@C.T@P@B) +K+ i*(mat_ones - 2*P)
            q=sinkhorn(ones_augm_, ones_, deriv[:n+1, :m], reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-5) 
            alpha = (2 / float(2 + it) )                                             
            P[:n, :m] = P[:n, :m] + alpha * (q[:n, :m] - P[:n, :m])
    P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
    forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
    result = P2@B.numpy()@P2.T
    forbnorm = LA.norm(A[:n,:n] - result[:n,:n], 'fro')**2
    return P, forbnorm,row_ind,col_ind
#used
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

##used
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

# used
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

# used
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


