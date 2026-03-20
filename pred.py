import numpy as np
import math
import torch
import sys
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sinkhorn import sinkhorn
from numpy import linalg as LA
import networkx as nx
import time
from multiprocessing import Pool
import warnings
import ot
#from memory_profiler import profile
from hung_utils import convertToPermHungarian2new,convertToPermHungarian,PermHungarian
from feature_util import feature_extraction,feature_extraction1
warnings.filterwarnings('ignore')
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)

def projection(P, lr, v1, v2, m, n, optimizer):
    if optimizer == 'sinkhorn':
        try:
            P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5, method='sinkhorn_log')
        except:
            P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5, method='sinkhorn_log')
    else:
        P_Sk = ot.emd(v1, v2, P.grad.data, numItermax = 100000, numThreads=30)
    P.grad.zero_()
    P.data = (1-lr)*P.data + lr*P_Sk
    return P

#used-all
def projectionN(P, lr, v1, v2,m,n,optimizer):
    if optimizer == 'sinkhorn':
        try:
            P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5)
        except:
            P_Sk = ot.sinkhorn(v1, v2, P.grad.data, 1.0, numItermax = 1500, stopThr = 1e-5, method='sinkhorn_log')
    else:
        P_Sk = ot.emd(v1, v2, P.grad.data, numItermax = 100000, numThreads=4)
    P.grad.zero_()
    P.data[:m,:n] = (1-lr)*P.data[:m,:n] + lr*P_Sk[:m,:n]
    return P

def enlargeMatrices(C,A1,size):
    new_node = max(C.nodes) + 1
    C.add_node(new_node)
    new_node = max(A1.nodes) + 1
    A1.add_node(new_node)
    return C,A1

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
    D = euclidean_distances(F1, F2)
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
    D = euclidean_distances(F1, F2)    
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
    D = euclidean_distances(F1, F2)
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
    feat = euclidean_distances(F1, F2)
    zeros_row = np.zeros((1, feat.shape[1]))
    feat=np.vstack([feat, zeros_row])
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = euclidean_distances(F1, F2)
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
    feat = euclidean_distances(F1, F2)
    zeros_row = np.zeros((1, feat.shape[1]))
    feat=np.vstack([feat, zeros_row])
        
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = euclidean_distances(F1, F2)
    D = torch.tensor(D, dtype = torch.float64)
    
    P, forbnorm,row_ind,col_ind = Alpine_pp_new_supervised(A[:n1,:n1], B,feat,mu*D,gtGq,gtGt, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm  

def AlpineTopk(A,B,k,t1,P1=None,beta=1.0):
    size=np.shape(A)[0]-np.shape(B)[0]
    C = B.copy()
    A1=A.copy()
    C,A1=enlargeMatrices(C,A1,size)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    
    F1 = feature_extraction1(A1)
    F2 = feature_extraction1(C)
    D = euclidean_distances(F2, F1)
    D = D[:np.shape(B)[0]+1,:np.shape(A)[0]]
    D=torch.tensor((D), dtype = torch.float64)
    P=Alpine_pp_new(B,A,D, 10, 10,1)
    P = P[:-1]
    P=P.detach().numpy()
    P3=np.multiply(P,1-P[-1].reshape(1,len(A)))
    _,row_ind,col_ind = PermHungarian(P)
    return P,P3,col_ind

def AlpineTorchTopk(A,B,k,t1,P1=None,beta=1.0):
    size=np.shape(A)[0]-np.shape(B)[0]
    C = B.copy()
    A1=A.copy()
    C,A1=enlargeMatrices(C,A1,size)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    F1 = feature_extraction1(A1)
    F2 = feature_extraction1(C)
    D = euclidean_distances(F2, F1)
    D = D[:np.shape(B)[0]+1,:np.shape(A)[0]]
    D=torch.tensor((D), dtype = torch.float64)
    P=Alpine_pp_torch(B,A,D, 10, 10,1)
    P=P[:-1]
    P=P.detach().numpy()
    P3=P
    _,row_ind,col_ind = PermHungarian(P)
    return P,P3,col_ind

def AlpinePlusTopk(A,B,k,t1,P1=None,beta=1.0):
    size=np.shape(A)[0]-np.shape(B)[0]
    C = B.copy()
    A1=A.copy()
    C,A1=enlargeMatrices(C,A1,size)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    
    F1 = feature_extraction1(A1)
    F2 = feature_extraction1(C)
    D = euclidean_distances(F2, F1)
    D = D[:np.shape(B)[0]+1,:np.shape(A)[0]+1]
    if(D[-1,-1]!=0):
        D[-1,-1]=0
    D=torch.tensor((D), dtype = torch.float64)
    P=partial_alignment(B,A,k,D,P1=P1)#        
    P1=P[:-1,:-1]
    P=P[:-1,:-1]
    P3=P1
    P=P.detach().numpy()
    P3=P3.detach().numpy()
    _,row_ind,col_ind = PermHungarian(P)
    return P,P3,col_ind

def Martian(A,B,k,t1,P1=None,beta=1.0):
    size=np.shape(A)[0]-np.shape(B)[0]
    C = B.copy()
    A1=A.copy()
    C,A1=enlargeMatrices(C,A1,size)
    A=torch.tensor((nx.to_numpy_array(A)), dtype = torch.float64)
    B=torch.tensor((nx.to_numpy_array(B)), dtype = torch.float64)
    
    F1 = feature_extraction1(A1)
    F2 = feature_extraction1(C)
    D = euclidean_distances(F2, F1)
    D = D[:np.shape(B)[0]+1,:np.shape(A)[0]+1]        
    D[-1,-1]=0
    D=torch.tensor((D), dtype = torch.float64)
    P=partial_alignmentPenalty(B,A,k,D,P1=P1,beta=beta)       
    P=P[:-1,:-1]
    P3 = P
    P=P.detach().numpy()
    P3=P3.detach().numpy()
    _,row_ind,col_ind = PermHungarian(P)
    return P,P3,col_ind

def Alpine_pp_torch(A,B, K, niter,A1,weight=1, optimizer='sinkhorn'):
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
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m
    for i in range(10):
        for it in range(1, 11):                                                                    
            loss =  torch.norm(A-I_p @ P @ B @ P.T @ I_p.T)**2-i*torch.trace(P[:m,:n].T@P[:m,:n])+torch.trace(P.T @ K[:m+1,:n])
            loss.backward()        
            lr = (2/float(2+it))       
            P.data = projectionN(P, lr, ones_augm_, ones_,m,n,optimizer) 
            P.requires_grad_(True)
    return P.data#

def partial_alignment(A, B, k, D,P1=None, optimizer='sinkhorn'):

    n = len(A)
    m = len(B)
    print('n, m', n, m)
    
    if P1 is None:
        P = torch.ones((n+1, m+1)).double()   
    #desired sum divided by number of elements
        P[:n, :m] *= k/(n*m)
        P[-1, :m] *= (m-k)/m
        P[:n, -1] *= (n-k)/n
        P[n, m] = 0
        P.data=P
    else:
        P = P1
    P.requires_grad_(True)
    v1 = torch.ones((n+1))
    v2 = torch.ones((m+1))
    v1[-1] = m-k
    v2[-1] = n-k

    for alpha in range(10):
        for t in range(1, 11, 1): 
            loss = torch.norm(torch.multiply(A, (1-P[:n, -1]).reshape(-1, 1)@((1-P[:n, -1]).reshape(-1, 1)).T) - P[:n, :m]@B@(P[:n, :m]).T)**2 - alpha*torch.trace(P[:n,:m].T@P[:n,:m])+torch.trace(P.T@D)
            loss.backward()
            lr = (2/float(2+t))       
            P.data = projection(P, lr, v1, v2, n, m, optimizer)   
            P.requires_grad_(True)
    return(P.data)    

def partial_alignmentPenalty(A, B, k, D, P1=None, optimizer='sinkhorn', beta=1.0):
    """
    Partial alignment with connectivity penalty using pairwise distances.
    A, B : adjacency matrices (torch.Tensor, n×n and m×m)
    k    : number of nodes to match
    D    : cost matrix (torch.Tensor, (n+1)×(m+1))
    """

    n = len(A)
    m = len(B)
    if P1 is None:
        P = torch.ones((n+1, m+1), dtype=torch.float64)
        P[:n, :m] *= k / (n * m)
        P[-1, :m] *= (m - k) / m
        P[:n, -1] *= (n - k) / n
        P[n, m] = 0.0
        P.requires_grad_(True)
    else:
        P = P1
        P.requires_grad_(True)

    # --- marginals for Sinkhorn projection ---
    v1 = torch.ones(n+1)
    v2 = torch.ones(m+1)
    v1[-1] = m - k
    v2[-1] = n - k
    beta=1
    # --- optimization loop ---
    for alpha in range(10):
        for t in range(1, 11):
            P_real = P[:n, :m]
            loss_align = torch.norm(A - P[:n, :m]@B@(P[:n, :m]).T)**2 - torch.sum(torch.multiply(A, torch.ones((n,1)).double()@((P[:n, -1].reshape(1, n))) + (torch.ones((n,1)).double()@((P[:n, -1].reshape(1, n)))).T - P[:n, -1].reshape(-1, 1)@(P[:n, -1].reshape(1, n))))

            y = 1 - P[:n, -1]     # shape (n,)
            z = 1 - P[-1, :m]     # shape (m,)
            y1=y.reshape(-1,1)
            z1=z.reshape(-1,1)
            loss_conn =  beta * ( y.T @ A @ y + z.T @ B @ z )
            loss = 2 * loss_align - loss_conn + torch.trace(P.T @ D)-alpha*torch.trace(P[:n,:m].T@P[:n,:m])
            loss.backward()            
            lr = 2.0 / float(2 + t)
            with torch.no_grad():
                P.data = projection(P, lr, v1, v2, n, m, optimizer)
            P.requires_grad_(True)
    return P.data

