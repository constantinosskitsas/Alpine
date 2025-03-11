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
warnings.filterwarnings('ignore')
os.environ["MKL_NUM_THREADS"] = "30"
torch.set_num_threads(30)
from help_functions import read_real_graph, read_list
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

def standardize_nodes(G):
    mapping = {node: str(node) for node in G.nodes()}
    return nx.relabel_nodes(G, mapping)
def plot(graph1, graph2):
    plt.figure(figsize=(12,4))
    plt.subplot(121)

    nx.draw(graph1)
    plt.subplot(122)

    nx.draw(graph2)
    plt.savefig('x1.png')

def feature_extraction1(G,simple = True):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""
    #G = standardize_nodes(G)
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
    #G = standardize_nodes(G)
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

def eucledian_dist2(F1, F2, D,maxp):
    for i in range(len(F1)):
        for j in range(len(F2)):
            if F1[i][0] > F2[j][0]:
                D[i][j] = D[i][j]*maxp
    return D
def eucledian_dist(F1, F2, n):
    #D = euclidean_distances(F1, F2)
    D = euclidean_distances(F1, F2)
    return D

def dist(A, B, P):
    obj = np.linalg.norm(np.dot(A, P) - np.dot(P, B))
    return obj*obj/2

def convex_init(A, B, D, mu, niter, n1):
    n = len(A)
    m = len(B)
    #P = torch.eye(n, dtype = torch.float64)
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    P = torch.ones((n,n), dtype = torch.float64)
    P = P/n
    #start = time.time()
    K=mu*D
    for i in range(niter):
        for it in range(1, 11):
            
            G = (torch.mm(torch.mm(A.T, A), P) - torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + torch.mm(torch.mm(P, B), B.T))/2 + mu*D + i*(mat_ones - 2*P)
            #G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K+ i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-03)
            
            #q = ot.sinkhorn(ones, ones, G, reg, numItermax = 1000, stopThr = 1e-5)

            alpha = 2.0 / float(2.0 + it)
            
            P = P + alpha * (q - P)
    #end = time.time()
    #print(end-start)
    P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
    P2 = torch.from_numpy(P2)
    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    return P, forbnorm,row_ind,col_ind



def convex_init1A(A, B, D, mu, niter):
    n = len(A)
    P = torch.eye(n, dtype = torch.float64)
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
def convex_init1(A, B, D, mu, niter):
    n = len(A)
    #P = torch.eye(n, dtype = torch.float64)
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

def convex_initSM(A, B, D, mu, niter):
    n = len(A)
    m = len(B)
    C_p = torch.zeros((m,m),dtype = torch.float64)
    for i in range(n):
        C_p[i,i] = 1
    C = C_p[:,:n]
    P = torch.ones((m,m), dtype = torch.float64)
    P = P/m
    A_p=C @ A @ C.T
    B_p=C @ B @ C.T
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D
    
    for i in range(niter):
        for it in range(1, 11):
            #G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K + i*(mat_ones - 2*P)
            G=C_p@P@B@B.T+C_p@P@B@B.T+ i*(mat_ones - 2*P)-C@A.T@C.T@P@B-C@A@C.T@P@B.T
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P
def convex_initQAP(A, B, niter):
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0 
    for i in range(1):
        for it in range(1, 11):
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T) + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

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
    return P, row_ind,col_ind

def convertToPermHungarian1112(M, n, m):
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

def convertToPerm(A, B, M, n1, n2):
    P_hung, ans_hung = convertToPermHungarian(M, n1, n2)
    P_greedy, ans_greedy = convertToPermGreedy(M, n1, n2)
    dist_hung = dist(A, B, P_hung)
    dist_greedy = dist(A, B, P_greedy)
    if dist_hung < dist_greedy:
        return P_hung, ans_hung
    else:
        return P_greedy, ans_greedy

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
    #D=eucledian_dist(F1,F2,n)    
    #D = torch.tensor(D, dtype = torch.float64)
    
    if(weight==0):
        D = torch.zeros((n,n),dtype = torch.float64)
    elif(weight==1):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
    elif(weight==1.5):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
        D = eucledian_dist2(F1,F2,D,1.5)
    elif(weight==2):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
        D = eucledian_dist2(F1,F2,D,2)
    else:
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)     
        D = euclidean_dist1(F1,F2,D,np.max(D))
    P, forbnorm,row_ind,col_ind = convex_init(A, B, D, mu, niter, n1)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    #_, ans = convertToPermHungarian2(P, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm

def predict_alignment(queries, targets, mu = 2, niter = 15):
    n = len(queries)
    mapping = []
    times = []
    for i in tqdm(range(n)):
        t1 = time.time()
        ans = align_new(queries[i], targets[i], mu, niter)
        mapping.append(ans)
        t2 = time.time()
        times.append(t2 - t1)
    return mapping, times

def Alpine_pp(A, B, K, niter,A1):

    n = len(A)
    m = len(B)
    C_p = torch.zeros((m,m),dtype = torch.float64)
    for i in range(n):
        C_p[i,i] = 1
    C = C_p[:,:n]
    P = torch.rand((n,m), dtype = torch.float64)
    #P = torch.ones((m,m), dtype = torch.float64)
    P = P/m
    ones = torch.ones(m, dtype = torch.float64)
    reg = 1.0
    mat_ones = torch.ones((m, m), dtype = torch.float64)

    ones_ = torch.ones(m, dtype = torch.float64)
    ones_augm_ = torch.ones(n+1, dtype = torch.float64)
    ones_augm_[-1] = m-n
    #mat_ones_np = mat_ones.numpy()
    #P_np = P.numpy()
    #P1 =K+ mat_ones_np - 2*P_np   
    #P2,_ = convertToPermHungarian(K*-1, m, n)
    #forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
    #return P1, forbnorm
    der=0
    sink=0
    proj=0
    start0 = time.time()
    #file_nodes = f'./data3_/multimanga_Noise15/50/0/nodes.txt'
    #array_2Acc = [[0 for _ in range(10)] for _ in range(10)]
    #array_2Frob = [[0 for _ in range(10)] for _ in range(10)]

    #Q_real = read_list(file_nodes)
    #K=torch.from_numpy(K)*0.1
    niter*10
    q=0
    for i in range(niter):
        #print()
        #curr=torch.trace((A - C.T@P@B@P.T@C).T@(A - C.T@P@B@P.T@C)) + torch.trace(K.T@P) - i*torch.trace(P.T@P)
        #print(curr.item(),end=" ")
        for it in range(1, 100):
            
            #start = time.time()
            #if (it<4):
            #    deriv=-torch.mm(torch.mm(A1.T, P), B)-torch.mm(torch.mm(A1, P), B.T)+ K+ i*(mat_ones - 2*P)
            #    q=sinkhorn(ones_, ones_, deriv, reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-5) 
            #    alpha = 2.0 / float(2.0 + it)                                               
            #    P = P + alpha * (q - P)
            #else:
            deriv = -2*C@A.T@C.T@P@B-2*C@A@C.T@P@B.T+2*C@(C.T@P@B@P.T@C@C.T@P@B.T+C.T@P@B.T@P.T@C@C.T@P@B) +K*0+ i*(mat_ones - 2*P)
            if it==1:
                print(deriv)
            #deriv = -2*C@A.T@C.T@P@B-2*C@A@C.T@P@B.T+2*C@(C.T@P@B@P.T@C@C.T@P@B.T+C.T@P@B.T@P.T@C@C.T@P@B) + i*(mat_ones - 2*P)               
            if it%10==1: q=sinkhorn(ones_augm_, ones_, deriv[:n+1, :m], reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-5) 
            alpha = (2 / float(2 + it) )                                             
            P[:n, :m] = P[:n, :m] + alpha * (q[:n, :m] - P[:n, :m])
            
            #q=sinkhorn_stabilized(ones_augm_, ones_, deriv[:n+1, :m], reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-5) 
            #end1 = time.time()
            #sink=sink+(end1-start1)
            #start2 = time.time()
                
            #curr=torch.trace((A - C.T@P@B@P.T@C).T@(A - C.T@P@B@P.T@C)) + torch.trace(K.T@P) - i*torch.trace(P.T@P)
            #print(curr.item(),end=" ")
            #end2 = time.time()
            #proj=proj+end2-start
            #P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
           # _, ans=convertToPermHungarian2new(row_ind,col_ind, n, m)
    #_, ans = convertToPermHungarian2(P, n1, n2)
            #list_of_nodes = []
            #for el in ans: list_of_nodes.append(el[1])
            #tempAcc= np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
            #forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
            #array_2Acc[i][it-1]=tempAcc
            #array_2Frob[i][it-1]=forbnorm
    #start2 = time.time()
    #K+mat_ones - 2*P
    print(P)
    P2,row_ind,col_ind = convertToPermHungarian(P, m, n)
    
    print(row_ind)
    print(col_ind)
    print(P2)
    #end2 = time.time()
    #end0 = time.time()
    #print("Derivative :", der)
    #print("Sinkhorn :" ,sink)
    #print("Projection : ",proj)
    #print("Hungarian: ",end2-start2)
    #print("All: ",end0-start0)
    #start2 = time.time()
    forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
    result = P2@B.numpy()@P2.T
    forbnorm = LA.norm(A[:n,:n] - result[:n,:n], 'fro')**2
    print("one")
    print(A[:n,:n])
    print("two")
    print(result[:n,:n])

    #end2 = time.time()
    #print("Frob :",end2-start2)

    #print("Reults ACC")
    #for row in array_2Acc:
    #    print(row)
    #print("Reults Frob")
    #for row in array_2Frob:
    #    print(row)
    return P, forbnorm,row_ind,col_ind


def Alpine_pp_new(A, B, K, niter,A1):

    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    Pi = torch.rand((m+1,n), dtype = torch.float64)
    Pi = Pi/n
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)

    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m

    for i in range(niter):
        for it in range(1, 10):

            #deriv = -2*I_p@A.T@I_p.T@P@B-2*I_p@A@I_p.T@P@B.T+2*I_p@(I_p.T@P@B@P.T@I_p@I_p.T@P@B.T+I_p.T@P@B.T@P.T@I_p@I_p.T@P@B) +K*0+ i*(mat_ones - 2*P)
            deriv=-4*I_p.T@(A-I_p@Pi@B@Pi.T@I_p.T)@I_p@Pi@B+K+ i*(mat_ones - 2*P)

            q=sinkhorn(ones_augm_, ones_, deriv, reg,method="sinkhorn",maxIter = 1500, stopThr = 1e-5) 
            alpha = (2 / float(2 + it) )                                             
            Pi = Pi + alpha * (q - Pi)
    print(Pi)
    P2,row_ind,col_ind = convertToPermHungarian(Pi, n, m)
    
    print(row_ind)
    print(col_ind)
    print(P2)

    forbnorm = LA.norm(A - I_p.T@P2@B@P2.T@I_p, 'fro')**2
    result = P2@B.numpy()@P2.T
    forbnorm = LA.norm(A[:n,:n] - result[:n,:n], 'fro')**2
    print("one")
    print(A[:n,:n])
    print("two")
    print(result[:n,:n])

    return P, forbnorm,row_ind,col_ind


def convex_initSM(A, B, K, niter):

    n = len(A)
    m = len(B)
    edges_left = torch.sum(B)-torch.sum(A)
    lamb=0#1/(10000*(int(edges_left)))
    C_p = torch.zeros((m,m),dtype = torch.float64)
    for i in range(n):
        C_p[i,i] = 1
    C = C_p[:,:n]
    P = torch.ones((m,m), dtype = torch.float64)
    P = P/m
    ones = torch.ones(m, dtype = torch.float64)
    reg = 1.0
    mat_ones = torch.ones((m, m), dtype = torch.float64)
    
    for i in range(niter):
        for it in range(1, 11):
            deriv = -2*C@A.T@C.T@P@B-2*C@A@C.T@P@B.T+2*C@(C.T@P@B@P.T@C@C.T@P@B.T+C.T@P@B.T@P.T@C@C.T@P@B) +K+ i*(mat_ones - 2*P)
            #diff = torch.trace((P@B@P.T-C_p.T@P@B@P.T@C_p)@((P@B@P.T-C_p.T@P@B@P.T@C_p).T))-edges_left
            #print(diff, 2*lamb*diff)
            #deriv += 2*lamb*diff*(
            #               +2*P@B@B.T@P.T
            #               -2*(C_p.T@B@P.T@C_p@P@B.T+C_p.T@P@B.T@P.T@C_p@P@B+C_p@P@B.T@P.T@C_p.T@P@B+C_p@P@B@P.T@C_p.T@P@B.T)
            #               +2*(C_p@C_p.T@P@B@P.T@C_p@C_p.T@P@B.T+C_p@C_p.T@P@B.T@P.T@C_p@C_p.T@P@B)
            #               )
            #print(+2*P@B@B.T@P.T
            #               -2*(C_p.T@B@P.T@C_p@P@B.T+C_p.T@P@B.T@P.T@C_p@P@B+C_p@P@B.T@P.T@C_p.T@P@B+C_p@P@B@P.T@C_p.T@P@B.T)
            #               +2*(C_p@C_p.T@P@B@P.T@C_p@C_p.T@P@B.T+C_p@C_p.T@P@B.T@P.T@C_p@C_p.T@P@B))
            #print(f'Derivative of forb. norm:\n{deriv}\n\n')
            q = sinkhorn(ones, ones, deriv, reg, numItermax = 1500, stopThr = 1e-3)
            #start = time.time()
            #q=m*ot.sinkhorn(ones/m, ones/m ,deriv, reg,method="sinkhorn", numItermax = 1500, stopThr = 1e-3)
            
            #end = time.time()
            #print(end - start)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
            #P[:n, :m] = P[:n, :m] + alpha * (q[:n, :m] - P[:n, :m])
            #print(f'perm. matrix in {it}th iteration:\n{P}')
    P2,_ = convertToPermHungarian(P, m, m)
    forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
    return P, forbnorm

def convex_initSM2(A, B, K, niter,weight):
    from numpy import linalg as LA
    n = len(A)
    m = len(B)
    edges_left = torch.sum(B)-torch.sum(A)
    lamb=0#1/(10000*(int(edges_left)))
    C_p = torch.zeros((m,m),dtype = torch.float64)
    for i in range(n):
        C_p[i,i] = 1
    C = C_p[:,:n]
    P = torch.ones((m,m), dtype = torch.float64)
    P = P/m
    ones = torch.ones(m, dtype = torch.float64)
    reg = 1.0
    mat_ones = torch.ones((m, m), dtype = torch.float64)
    for i in range(niter):
        for it in range(1, 11):
            deriv = -2*C@A.T@C.T@P@B-2*C@A@C.T@P@B.T+2*C@(C.T@P@B@P.T@C@C.T@P@B.T+C.T@P@B.T@P.T@C@C.T@P@B) + K + i*(mat_ones - 2*P)
            #deriv += weight*((P@B@P.T)*C_p - C@A@C.T)@(P@B)
            
            
            #diff = torch.trace((P@B@P.T-C_p.T@P@B@P.T@C_p)@((P@B@P.T-C_p.T@P@B@P.T@C_p).T))-edges_left
            #print(diff, 2*lamb*diff)
            #deriv += 2*lamb*diff*(
            #               +2*P@B@B.T@P.T
            #               -2*(C_p.T@B@P.T@C_p@P@B.T+C_p.T@P@B.T@P.T@C_p@P@B+C_p@P@B.T@P.T@C_p.T@P@B+C_p@P@B@P.T@C_p.T@P@B.T)
            #               +2*(C_p@C_p.T@P@B@P.T@C_p@C_p.T@P@B.T+C_p@C_p.T@P@B.T@P.T@C_p@C_p.T@P@B)
            #               )
            #print(+2*P@B@B.T@P.T
            #               -2*(C_p.T@B@P.T@C_p@P@B.T+C_p.T@P@B.T@P.T@C_p@P@B+C_p@P@B.T@P.T@C_p.T@P@B+C_p@P@B@P.T@C_p.T@P@B.T)
            #               +2*(C_p@C_p.T@P@B@P.T@C_p@C_p.T@P@B.T+C_p@C_p.T@P@B.T@P.T@C_p@C_p.T@P@B))
            #print(f'Derivative of forb. norm:\n{deriv}\n\n')
            q = sinkhorn(ones, ones, deriv, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
            #print(f'perm. matrix in {it}th iteration:\n{P}')
    #P[:, :n] *= weight
    #P[:, n:m] *= (1-weight)
    P2,_ = convertToPermHungarian(P, m, m)
    forbnorm = LA.norm(A - C.T@P2@B@P2.T@C, 'fro')**2
    return P, forbnorm

def Alpine(Gq, Gt, mu=1, niter=10, weight=2):
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
    mu=0.1     
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
    D[-1:]*=0
    print((D))

    #P, forbnorm,row_ind,col_ind = Alpine_pp(A[:n1,:n1], B, mu*D, niter)
   # P, forbnorm,row_ind,col_ind = Alpine_pp(A[:n1,:n1], B, mu*D, niter,A)
    P, forbnorm,row_ind,col_ind = Alpine_pp_new(A[:n1,:n1], B, mu*D, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
   # _, ans = convertToPermHungarian2(P, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm    


def Fugal_pp(A, B, D, mu, niter, n1):
    n = len(A)
    m = len(B)
    #P = torch.ones((n,n), dtype = torch.float64)
    P = torch.rand((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D*1
    #P=sinkhorn(ones, ones, K, reg, maxIter = 1500, stopThr = 1e-3)
    #P=torch.zeros((n,n), dtype = torch.float64)
    for i in range(niter):
        for it in range(1, 11):
            #G=  A.T@torch.sign(A @ P- P@B)- torch.sign(A@P-P@B) @ B.T+K+ i*( - 2*P)
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)#+ K*0+ i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 1500, stopThr = 1e-5)
            if (it==5):
                print(G)
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
    #F1 = feature_extraction1(Gq)
    #F2 = feature_extraction1(Gt)
    D = eucledian_dist(F1,F2,n)
    #print(D)
    P, forbnorm,row_ind,col_ind = Fugal_pp(A, B, mu,D, niter,n1)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
   # _, ans = convertToPermHungarian2(P, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm 







def align_SM(Gq, Gt, mu=1, niter=10, weight=1.0):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    if(weight==0):
        D = torch.zeros((n,n),dtype = torch.float64)
    elif(weight==1):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
    elif(weight==1.5):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
        D = eucledian_dist2(F1,F2,D,1.5)
    elif(weight==2):
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)
        D = eucledian_dist2(F1,F2,D,2)
    else:
        D = torch.zeros((n,n),dtype = torch.float64)
        D = eucledian_dist(F1,F2,n)     
        D = euclidean_dist1(F1,F2,D,np.max(D))
    #if weight==0:
    P, forbnorm = convex_initSM(A[:n1,:n1], B, mu*D, niter)
    #else: 
    #    P, forbnorm = convex_initSM2(A[:n1,:n1], B, mu*D, niter,weight)
    _, ans = convertToPermHungarian(P, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm
