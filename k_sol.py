import torch
import numpy as np
import copy
import ot
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import torch
from sklearn.metrics.pairwise import euclidean_distances
import ot
import scipy
from numpy import linalg as LA
from collections import Counter
def feature_extraction(G,simple):
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
def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    return row_ind,col_ind

def projection(P, lr, v1, v2, m, n):
    P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5)
    P.grad.zero_()
    P.data = (1-lr)*P.data + lr*P_Sk
    #P.data[m,n] = 0
    return P
    
def projectionN(P, lr, v1, v2,m,n):
    P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5)
    row_sums = P_Sk.sum(dim=1).double()
    col_sums = P_Sk.sum(dim=0).double()
    v1=v1.double()
    v2=v2.double()
    if (not(torch.allclose(row_sums, v1, atol=1e-2)) and (torch.allclose(col_sums, v2, atol=1e-2))):
        print("Error in projection")
    #update P
    P.grad.zero_()
    P.data[:m,:n] = (1-lr)*P.data[:m,:n] + lr*P_Sk[:m,:n]
    return P

def partial_alignmentDD(A, B, k,D):
    print(f'Dimentions of A is {A.shape}')
    print(f'Dimentions of A is {B.shape}')
    
    #sizes 
    n = len(A)
    m = len(B)
    P = torch.ones((n+1, m+1)).double()
    
    #desired sum divided by number of elements
    P[:n, :m] *= k/(n*m)
    P[-1, :m] *= (m-k)/m
    P[:n, -1] *= (n-k)/n
    P[:-1,:] *= 1/n
    P[-1,:] *= (n-m)/n
    P[n, m] = 0
    
    P.requires_grad_(True)
    v1 = torch.ones((n+1))
    v2 = torch.ones((m+1))
    v1[-1] = m-k
    v2[-1] = n-k
    for alpha in range(10):
        for t in range(1, 10, 1):                                                
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n, :m].T@P[:n, :m])+torch.trace(P[:n+1, :m+1].T@D)
            loss.backward()
            lr = (2/float(2+(t/1)) )        
            P.data = projection(P, lr, v1, v2,n,m)   
            P.requires_grad_(True)
    
    for alpha in range(10):
        for t in range(1, 10, 1):                                                
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n, :m].T@P[:n, :m])+torch.trace(P[:n+1, :m+1].T@D)
            loss.backward()
            lr = (2/float(2+(t/1)) )        
            P.data = projection(P, lr, v1, v2,n,m)   
            P.requires_grad_(True)
    
    return(P.data) 
def partial_alignmentGrad(A, B, k,D):
    print(f'Dimentions of A is {A.shape}')
    print(f'Dimentions of A is {B.shape}')
    
    #sizes 
    n = len(A)
    m = len(B)
    P = torch.ones((n+1, m+1)).double()
    k1=k
    k=n
    #desired sum divided by number of elements
    P[:n, :m] *= k/(n*m)
    P[-1, :m] *= (m-k)/m
    P[:n, -1] *= (n-k)/n
    P[:-1,:] *= 1/n
    P[-1,:] *= (n-m)/n
    P[n, m] = 0
    
    P.requires_grad_(True)
    v1 = torch.ones((n+1))
    v2 = torch.ones((m+1))
    v1[-1] = m-k
    v2[-1] = n-k
    for alpha in range(10):
        for t in range(1, 10, 1):                                                
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n, :m].T@P[:n, :m])+torch.trace(P[:n+1, :m+1].T@D)
            loss.backward()
            lr = (2/float(2+(t/1)) )        
            P.data = projection(P, lr, v1, v2,n,m)   
            P.requires_grad_(True)
    difnum=k-k1
    step = difnum // 10  # Integer step size
    remainder = difnum % 10  # Remaining difference
    for alpha in range(10):
        if alpha == 9:  # Last step, ensure exact k1
            k = k1
        else:
            k -= step
        if alpha < remainder:  # Distribute remainder in early steps
            k -= 1  
        v1[-1] = m-k
        v2[-1] = n-k
        for t in range(1, 10, 1):                                                
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n, :m].T@P[:n, :m])+torch.trace(P[:n+1, :m+1].T@D)
            loss.backward()
            lr = (2/float(2+(t/1)) )        
            P.data = projection(P, lr, v1, v2,n,m)   
            P.requires_grad_(True)
    
    return(P.data)    

def partial_alignment(A, B, k, D):

    
    #sizes 
    n = len(A)
    m = len(B)
    P = torch.ones((n+1, m+1)).double()
    
    #desired sum divided by number of elements
    P[:n, :m] *= k/(n*m)
    P[-1, :m] *= (m-k)/m
    P[:n, -1] *= (n-k)/n
    #P[:-1,:] *= 1/n
    #P[-1,:] *= (n-m)/n
    P[n, m] = 0
    
    P.requires_grad_(True)
    v1 = torch.ones((n+1))
    v2 = torch.ones((m+1))
    v1[-1] = m-k
    v2[-1] = n-k

    #
    #k1=n
    #v1[-1] = m-k1
    #v2[-1] = n-k1
    #numA=(n-k)//10
    for alpha in range(10):
        for t in range(1, 11, 1):                                                
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n, :m].T@P[:n, :m])+torch.trace(P.T@D)
            #loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P.T@P)+torch.trace(P.T@D)

            loss.backward()
            lr = (2/float(2+t))       
            P.data = projection(P, lr, v1, v2, n, m)   
            P.requires_grad_(True)
    return(P.data)            
            
def read_real_graph(n, name_):
    print(f'Making {name_} graph...')
    filename = open(f'{name_}', 'r')
    lines = filename.readlines()
    G = nx.Graph()
    for i in range(n): G.add_node(i)
    for line in lines:
        u_v = (line[:-1].split(' '))
        u = int(u_v[0])
        v = int(u_v[1])
        G.add_edge(u, v)
    print(f'Done {name_} Peter...')
    return G 

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
            q = ot.sinkhorn(ones_augm_, ones_, deriv, 1.0, numItermax = 1500, stopThr = 1e-5)
            alpha = (2 / float(2 + it) )    
            Pi[:m,:n] = Pi[:m,:n] + alpha * (q[:m,:n] - Pi[:m,:n])
    return Pi#, forbnorm,row_ind,col_ind

def Alpine_pp_torch(A,B, K, niter,A1,weight=1):
    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    P=torch.ones((m+1,n),dtype = torch.float64)
    P[:-1,:] *= 1/n
    P[-1,:] *= (n-m)/n
    P.requires_grad_(True)
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m
    #D=K[:m+1,:n]
    for i in range(10):
        for it in range(1, 11):
            loss =  torch.norm(A-I_p @ P @ B @ P.T @ I_p.T)**2-i*torch.trace(P.T@P)+torch.trace(P.T @ K[:m+1,:n])
            loss.backward()        
            #frank-wolfe 
            lr = (2/float(2+it) )       
            P.data = projectionN(P, lr, ones_augm_, ones_,m,n) 
            P.requires_grad_(True)
    return P.data#

def enlargeMatrices(C,A1,size):

    new_node = max(C.nodes) + 1
    C.add_node(new_node)
    #C.add_edge(new_node,new_node)
    #for _ in range(size):
    #    new_node = max(C.nodes) + 1  # Assign the next available node ID
    #    C.add_node(new_node)
    #    C.add_edge(new_node,new_node)
    new_node = max(A1.nodes) + 1
    A1.add_node(new_node)
    #A1.add_edge(new_node,new_node)
    return C,A1
def Aeval(A,B,P,k):
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    row_ind,col_ind = convertToPermHungarian(P, 1, 1)
    selected_pairs = list(zip(row_ind, col_ind))
    
    # Get the total cost for each pair
    pair_costs = P[row_ind, col_ind]
        
    sorted_pairs = sorted(zip(selected_pairs, pair_costs), key=lambda x: x[1],reverse=True)  # Sort by cost
    top_n_pairs = sorted_pairs[:k]  # Select top n
    top_n_assignments = [(pair, cost) for pair, cost in top_n_pairs]
    top_n_row_indices = [pair[0][0] for pair in top_n_pairs]
    top_n_col_indices = [pair[0][1] for pair in top_n_pairs]

    A_sub = A[top_n_col_indices, :][:, top_n_col_indices]
    B_sub = B[top_n_row_indices, :][:, top_n_row_indices]
    #np.savetxt("A_sub.csv", A_sub, delimiter=",", fmt="%.5f")
    #np.savetxt("B_sub.csv", B_sub, delimiter=",", fmt="%.5f")
    # Now compute the Frobenius norm of the difference between the two submatrices.
    fro_norm_diff = torch.norm(A_sub - B_sub, p='fro').item()
    print(fro_norm_diff)

    #print(torch.norm(A[col_ind,:][:,col_ind] - B[row_ind,:][:,row_ind], p='fro'))
    return fro_norm_diff
def topkEval(ORIG,A,B,P,k):
    ORIG=torch.tensor((nx.to_numpy_array(ORIG)),dtype=torch.float64)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    row_ind,col_ind = convertToPermHungarian(P, 1, 1)
    selected_pairs = list(zip(row_ind, col_ind))
    
    # Get the total cost for each pair
    pair_costs = P[row_ind, col_ind]
        
    sorted_pairs = sorted(zip(selected_pairs, pair_costs), key=lambda x: x[1],reverse=True)  # Sort by cost
    top_n_pairs = sorted_pairs[:k]  # Select top n
    top_n_assignments = [(pair, cost) for pair, cost in top_n_pairs]
    top_n_row_indices = [pair[0][0] for pair in top_n_pairs]
    top_n_col_indices = [pair[0][1] for pair in top_n_pairs]

    A_sub = A[top_n_col_indices, :][:, top_n_col_indices]
    B_sub = B[top_n_row_indices, :][:, top_n_row_indices]
    #np.savetxt("A_sub.csv", A_sub, delimiter=",", fmt="%.5f")
    #np.savetxt("B_sub.csv", B_sub, delimiter=",", fmt="%.5f")
    # Now compute the Frobenius norm of the difference between the two submatrices.
    fro_norm_diff = torch.norm(A_sub - B_sub, p='fro').item()
    #FA=torch.norm(A_sub - ORIG, p='fro').item()
    #FB=torch.norm(B_sub - ORIG, p='fro').item()
    FA=0
    FB=0
    print(fro_norm_diff,FA,FB)

    #print(torch.norm(A[col_ind,:][:,col_ind] - B[row_ind,:][:,row_ind], p='fro'))
    return fro_norm_diff,FA,FB
def kPMatch(A,B,k,t1):
    size=np.shape(A)[0]-np.shape(B)[0]
    C = B.copy()
    A1=A.copy()
    C,A1=enlargeMatrices(C,A1,size)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    
    F1 = feature_extraction(A1,True)
    F2 = feature_extraction(C,True)
    
    if (t1==0):
        D = euclidean_distances(F2, F1)
        D = D[:np.shape(B)[0]+1,:np.shape(A)[0]]
        D=torch.tensor((D), dtype = torch.float64)
        P=Alpine_pp_new(B,A,D, 10, 10,1)
        P = P[:-1]
    elif(t1==1):
        D = euclidean_distances(F2, F1)
        D = D[:np.shape(B)[0]+1,:np.shape(A)[0]]
        D=torch.tensor((D), dtype = torch.float64)
        P=Alpine_pp_torch(B,A,D, 10, 10,1)
        P3=np.multiply(P,1-P[-1].reshape(1,len(A)))
        P = P[:-1]
        P=P.detach().numpy()
        P3=P3.detach().numpy()
        P3=P
    elif(t1==3):
        D = euclidean_distances(F2, F1)
        #D = D[:np.shape(B)[0]+1,:np.shape(A)[0]+1]
        D[-1,-1]=0
        D=torch.tensor((D), dtype = torch.float64)
        P=partial_alignment(B,A,k,D)#
        #P=partial_alignmentDD(B,A,k,D)#
        
        P1=P[:-1,:-1]
        P2=np.multiply(P1,1-P[:-1,-1].reshape(len(B),1))
        P3=np.multiply(P2,1-P[-1,:-1].reshape(1,len(A)))
        P3 = P[:-1,:-1]
        P=P[:-1,:-1]
        P=P.detach().numpy()
        P3=P3.detach().numpy()
        print(np.shape(P))
        print(np.shape(P1))
    else :
        D = euclidean_distances(F2, F1)
        D = D[:np.shape(B)[0]+1,:np.shape(A)[0]+1]
        D=torch.tensor((D), dtype = torch.float64)
        P=partial_alignment(B,A,k,D)#
        #P=partial_alignmentGrad(B,A,k,D)#
        P = P[:-1,:-1]
        P=P.detach().numpy()
    row_ind,col_ind = convertToPermHungarian(P, 1, 1)
    return P,P3,col_ind



