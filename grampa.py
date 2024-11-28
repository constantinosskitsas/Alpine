import numpy as np
#from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv
from numpy.linalg import eigh,eig
import networkx as nx 
import random
from math import floor, log2
#from lapsolver import solve_dense
import scipy as sci
#from lapsolver import solve_dense
from numpy import inf, nan
import scipy.sparse as sps
import math
import os
import scipy
from numpy import linalg as LA
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
def Grampa(Gq,Gt):
    print("Grampa")
    #os.environ["MKL_NUM_THREADS"] = "10"
    eta=0.2
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    nmin= min(n1,n2)
    for i in range(n1, n):
        Gq.add_node(i)
        Gq.add_edge(i,i)
    for i in range(n2, n):
        Gt.add_node(i)
    A = nx.to_numpy_array(Gq)
    B = nx.to_numpy_array(Gt)

    l,U =eigh(A)
    mu,V = eigh(B)
    
    l = np.array([l])
    mu = np.array([mu])
    dtype = np.float32
  #Eq.4
    coeff = 1.0/((l.T - mu)**2 + eta**2)
  #Eq. 3

    coeff = coeff * (U.T @ np.ones((n,n)) @ V)
    
  
  #coeff = coeff * (U.T @ K @ V)
    X = U @ coeff @ V.T
    Xt = X.T
    Xt=X*1
    P2,_ = convertToPermHungarian(Xt, n, n)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian(Xt, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm