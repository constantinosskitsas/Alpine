
import numpy as np
import sklearn.metrics.pairwise
import scipy.sparse as sps
import argparse
import time
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
# from data import ReadFile
import unsup_align, embedding
import networkx as nx
from numpy import linalg as LA
import scipy
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
def align_embeddings(embed1, embed2, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
        corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=10, reg=1, P=corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(
            embed1, embed2, apply_sqrt=False, niter=10, reg=1, P=corr)

    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=1, bsz=10, nepoch=5, niter=10, reg=0.05)
 
    aligned_embed1 = embed1.dot(dim_align_matrix)

    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric='euclidean', num_top=10)

    return alignment_matrix, sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
def align_embeddings1(embed1, embed2, CONE_args, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
        corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=10, reg=1, P=corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(
            embed1, embed2, apply_sqrt=False, niter=10, reg=1, P=corr)
    # print(corr_mat)
    # print(np.max(corr_mat, axis=0))
    # print(np.max(corr_mat, axis=1))

    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=1, bsz=10, nepoch=5, niter=10, reg=0.05)
    # print(dim_align_matrix.shape, corr_mat.shape)

    # Step 3: Match Nodes with Similar Embeddings
    # Align embedding spaces
    aligned_embed1 = embed1.dot(dim_align_matrix)

    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric='euclidean', num_top=10)
    return alignment_matrix, sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)



def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    # print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()

def coneGAM(Gq,Gt):
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
    dim= 128
    if (dim>=n1):
        dim= n1-10
    emb_matrixA = embedding.netmf(
        A, dim=dim, window=10, b=1, normalize=True)

    emb_matrixB = embedding.netmf(
        B, dim=dim, window=10, b=1, normalize=True)
    alignment_matrix, cost_matrix = align_embeddings(
        emb_matrixA,
        emb_matrixB,
        adj1=csr_matrix(A),
        adj2=csr_matrix(B),
        struc_embed=None,
        struc_embed2=None
    )
    cost_matrix=cost_matrix*-1
    P2,_ = convertToPermHungarian(cost_matrix, n, n)

    forbnorm = LA.norm(A[:n1,:n1] - (P2@B@P2.T)[:n1,:n1], 'fro')**2
    P_perm,ans = convertToPermHungarian(cost_matrix, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm



def main(data, **args):
    print("Cone")
    Src = data['Src']
    Tar = data['Tar']

    if args['dim'] > Src.shape[0] - 1:
        args['dim'] = Src.shape[0] - 1
    emb_matrixA = embedding.netmf(
        Src, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)

    emb_matrixB = embedding.netmf(
        Tar, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    alignment_matrix, cost_matrix = align_embeddings(
        emb_matrixA,
        emb_matrixB,
        args,
        adj1=csr_matrix(Src),
        adj2=csr_matrix(Tar),
        struc_embed=None,
        struc_embed2=None
    )

    return alignment_matrix, cost_matrix
