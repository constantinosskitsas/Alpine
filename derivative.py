import numpy as np
from pred import convex_initSM
import torch
import scipy


# Matrix A (Adjacency of a 3-clique)
A = np.array([[0, 0, 1],
              [0, 0, 1],
              [1, 1, 0]])

# Matrix B (Adjacency of a 5-clique)
B = np.array([[0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0]])

# Matrix C (5x3 matrix with diagonal elements equal to 1)
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 0]])

# Matrix P (Permutation matrix)
P = np.array([[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1]])

P = np.ones((len(B),len(B)))
P = P/len(B)


print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nMatrix C:")
print(C)

print("\nMatrix P:")
print(P)


print("\nMatrix PBP.T:")
print(P@B@P.T)

print("\nMatrix C.TPBP.TC:")
print(C.T@P@B@P.T@C)

print("\nMatrix CAC.T")
print(C@A@C.T)


print("\nDerivative equals:")
print(-2*(C@(A.T@C.T@P@B+A@C.T@P@B-
             C.T@P@B@P.T@C@C.T@P@B.T-
             C.T@P@B.T@P.T@C@C.T@P@B)))

print()
print()

P_new = convex_initSM(torch.tensor(A,dtype = torch.float64), torch.tensor(B,dtype = torch.float64))

cols, rows = scipy.optimize.linear_sum_assignment(-P_new)

print(cols)
print(rows)