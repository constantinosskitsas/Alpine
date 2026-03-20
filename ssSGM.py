import numpy as np
from numpy import linalg
import scipy
from scipy.optimize import linear_sum_assignment
import random
import time

def Hungarian(Q, q):
    n1 = Q.shape[0]
    n2 = Q.shape[1]
    lb = Q.min()-1
    ub = Q.max()+1
    Q1 = np.hstack((Q,ub*np.ones((n1, n1-q))))
    Q2 = np.hstack((ub*np.ones((n2-q,n2)),lb*np.ones((n2-q,n1-q))))
    Q_sq = np.vstack((Q1, Q2))
    row_ind, col_ind = linear_sum_assignment(-Q_sq) # linear assignment
    X = np.zeros((n1+n2-q, n1+n2-q))
    for i in range (n1+n2-q):
        X[i,col_ind[i]] = 1
    Xstar = X[:n1,:n2]
    return (Xstar)

def sgm11(A,B,s,max_iteration,tol,k):
    m = A.shape[0]
    n = B.shape[0]
    #A11 = A[:s,:s]
    A22 = A[s:,s:]
    A21 = A[s:,:s]
    A12 = A[:s,s:]

    #B11 = B[:s,:s]
    B22 = B[s:,s:]
    B21 = B[s:,:s]
    B12 = B[:s,s:]

    run = 1
    current_iteration = 0
    Z = np.ones((m-s, n-s))*((k-s)/((m-s)*(n-s)))

    while (run == 1) and (current_iteration < max_iteration):
        
        f = (linalg.norm(np.matmul(A22 , Z) -  np.matmul(Z , B22)))**2
        current_iteration = current_iteration + 1
        M = np.matmul(A21, B12) + np.matmul(np.matmul(A22, Z), B22)
        Xstar = Hungarian(M, k-s)
        
        c = np.trace(np.matmul(np.matmul(A22, Xstar), np.matmul(B22, Xstar.T)))
        d = np.trace(np.matmul(np.matmul(A22, Z), np.matmul(B22, Z.T)))
        e = 2*np.trace(np.matmul(np.matmul(A22, Xstar), np.matmul(B22, Z.T)))

        u = 2*np.trace(np.matmul(np.matmul(A12, Xstar-Z), B21))

        if (c+d-e) != 0:
            alpha = - (-2*d + e + u)/2/(c + d - e)
        else:
            alpha = 0
        
        f0 = 0
        f1 = c - d + u
        falpha = (c + d - e) * alpha**2 + (-2*d + e + u) * alpha

        if (alpha > 0) and (alpha < 1) and (falpha > f1) and (falpha > f0):
            Z_next = alpha*Xstar + (1-alpha)*Z
        elif (f0>f1):
            Z_next = Z
        else:
            Z_next = Xstar
        fnew = (linalg.norm(np.matmul(A22 , Z_next) -  np.matmul(Z_next , B22)))**2
        
        if abs(fnew-f) > tol and fnew > tol and np.sum(abs(Z-Z_next)) > tol:
            run = 1
        else:
            run = 0
        Z = Z_next
    
    Z_temp = Hungarian(Z, k-s)

    PP1 = np.hstack((np.eye(s), np.zeros((s,n-s))))
    PP2 = np.hstack((np.zeros((m-s,s)), Z_temp))
    PP = np.vstack((PP1,PP2))

    return PP

def ssSGM_simulation(A, B, s, k, max_iteration, tol):
    A_ssSGM = 2 * A - 1
    B_ssSGM = 2 * B - 1
    A_ssSGM = A_ssSGM - np.diag(np.diag(A_ssSGM)) - np.eye(A_ssSGM.shape[0])
    B_ssSGM = B_ssSGM - np.diag(np.diag(B_ssSGM)) - np.eye(B_ssSGM.shape[0])
    PP = sgm11(A_ssSGM, B_ssSGM, s, max_iteration, tol, k)
    return PP