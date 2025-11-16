import networkx as nx
import torch
import numpy as np
from pred import eucledian_dist,feature_extraction1,feature_extraction,convertToPermHungarian
from pred import convertToPermHungarian2new
import ot
import scipy
from numpy import linalg as LA
from GradP.gradp import gradPMain
import time
import os
from help_functions import read_graph
from help_functions import read_real_graph, read_list
from resultsfolder import generate_new_id,create_new_folder,get_max_previous_id 
import pandas as pd
from JOENA.Joena_main import JOENA
from SlotaAlign.SlotaAlign_main import SlotaA
from NextAlign.NextA import NextAlign
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)
folderall = 'data3_'
experimental_folder=f'./{folderall}/res/'
new_id = generate_new_id(get_max_previous_id(experimental_folder))
experimental_folder=f'./{folderall}/res/_{new_id}/'   
def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Time:', time_diff)
    print()     
def PermHungarian(M):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    return _ ,row_ind,col_ind
def synchronize_features(F1, F2, anchors_G, anchors_G_Q, direction="F1_to_F2"):
    """
    Synchronize features along ground truth anchors.
    
    Parameters
    ----------
    F1 : np.ndarray, shape (n1, k)
        Feature matrix of graph G
    F2 : np.ndarray, shape (n2, k)
        Feature matrix of graph G_Q
    anchors_G : list[int]
        Anchor node indices in G (for F1)
    anchors_G_Q : list[int]
        Corresponding anchor node indices in G_Q (for F2)
    direction : str
        Either "F1_to_F2" or "F2_to_F1" indicating copy direction
    
    Returns
    -------
    F1_new, F2_new : np.ndarray
        Feature matrices after synchronization
    """
    # Make copies
    F1_new = F1.copy()
    F2_new = F2.copy()

    if direction == "F1_to_F2":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F2_new[u_q] = F1_new[u]
    elif direction == "F2_to_F1":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F1_new[u] = F2_new[u_q]
    else:
        raise ValueError("direction must be 'F1_to_F2' or 'F2_to_F1'")
    
    return F1_new, F2_new
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
    #SimOH=one_hop_similarity_matrix(A,B,gtA,gtB)
    #SimOH1=one_hop_similarity(A,B,gtA,gtB)
    #Sim2H=two_hop_similarity_matrix(A,B,gtA,gtB)
    #cost1H=1-SimOH
    #cost2H=1-Sim2H
    #costGT=cost1H#+cost2H
    #costGT=costGT*5
    costGT=one_hop_cost_neighbors(A,B,gtA,gtB)
    costGT=costGT+two_hop_cost_neighbors(A,B,gtA,gtB)
    dummy_row = torch.zeros((1, costGT.shape[1]), dtype=costGT.dtype, device=costGT.device)
    costGT = torch.cat([costGT, dummy_row], dim=0)
    #diff = torch.abs(SimOH - SimOH1)
    print("Here")
    #if torch.any(diff > 0.001):
    #    print("Matrices differ")
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
        costGT[i,j]=0
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
    A0 = np.mean(np.abs(feat))
    for outer in range(10):
        for it in range(1, 11):
            deriv= (-4*I_p.T @ (A - I_p @ Pi @ B @ Pi.T @ I_p.T) @ I_p @ Pi @ B)*dd + outer * (mat_ones - 2 * Pi) + K*2
            S0 = deriv.abs().mean().item()  # magnitude of structural gradient
            gamma_a = gamma * S0 / (A0 + 1e-4)
            deriv = deriv + gamma_a * (feat)+costGT*1
            #print(np.max(gamma_a*feat))
            deriv=deriv/5
            q=ot.sinkhorn(ones_augm_, ones_, deriv, 1.0, numItermax = 1500, stopThr = 1e-9)
            
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
def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end='\t')
    return G1, G2
def one_hop_similarity(A, B, gtA, gtB):
    """
    Loop-based 1-hop similarity using ground-truth anchor mapping.

    Args:
        A, B: adjacency matrices (torch.Tensor)
        gtA, gtB: lists of ground-truth anchor indices

    Returns:
        sim: similarity matrix (nA x nB)
    """
    nA, nB = A.shape[0], B.shape[0]

    # Convert to Python ints for safe dict operations
    gtA = [int(x) for x in gtA]
    gtB = [int(x) for x in gtB]

    # Ground-truth anchor mapping
    anchor_map = {a: b for a, b in zip(gtA, gtB)}
    anchor_values = set(anchor_map.values())

    # Precompute neighbors as Python ints
    neighA = [list(map(int, torch.nonzero(A[i]).view(-1).tolist())) for i in range(nA)]
    neighB = [set(map(int, torch.nonzero(B[j]).view(-1).tolist())) for j in range(nB)]

    # Precompute degrees of mappable neighbors
    degA_gt = [sum(1 for u in neighA[i] if u in anchor_map) for i in range(nA)]
    degB_gt = [sum(1 for v in neighB[j] if v in anchor_values) for j in range(nB)]

    # Initialize similarity matrix
    sim = torch.zeros((nA, nB), dtype=torch.float64)

    # Compute similarity
    for i in range(nA):
        for j in range(nB):
            count = 0
            for u in neighA[i]:
                if u in anchor_map:
                    v = anchor_map[u]
                    if v in neighB[j]:
                        count += 1

            denom = degA_gt[i] + degB_gt[j]
            sim[i, j] = 1.0 if denom == 0 else 2 * count / denom

    return sim
def two_hop_cost_neighbors(A, B, gtA, gtB):
    """
    Compute 2-hop cost based on ground-truth anchored neighbors.

    A, B : adjacency matrices (torch.Tensor)
    gtA, gtB : lists/arrays of ground-truth anchors
    Returns
    -------
    cost2H : (nA x nB) tensor of 2-hop costs
    """
    nA, nB = A.shape[0], B.shape[0]
    device = A.device

    # Compute 2-hop adjacency (binary)
    A2 = ((A + A @ A) > 0).double()  # include 1-hop edges as well
    B2 = ((B + B @ B) > 0).double()

    # Anchor mapping matrix
    M = torch.zeros((nA, nB), dtype=torch.float64, device=device)
    M[gtA, gtB] = 1.0

    # Count matched anchored neighbors (numerator)
    C2 = torch.mm(A2, torch.mm(M, B2.T))

    # Degrees: total number of anchored neighbors (denominator)
    maskA = torch.zeros(nA, dtype=torch.float64, device=device)
    maskA[gtA] = 1.0
    degA2_gt = A2 @ maskA.unsqueeze(1)

    maskB = torch.zeros(nB, dtype=torch.float64, device=device)
    maskB[gtB] = 1.0
    degB2_gt = B2 @ maskB.unsqueeze(1)

    denom2 = degA2_gt + degB2_gt.T

    # Compute cost: fraction of mismatched anchored neighbors
    cost2 = torch.zeros_like(C2)
    nonzero = denom2 != 0
    cost2[nonzero] = (degA2_gt + degB2_gt.T - 2*C2)[nonzero] / denom2[nonzero]
    w = torch.log1p(denom2) / torch.log1p(denom2.max())
    
    cost2[nonzero] = w[nonzero] * (degA2_gt + degB2_gt.T - 2*C2)[nonzero] / denom2[nonzero]
    
    # If both nodes have no anchored neighbors, cost = 0
    cost2[denom2 == 0] = 0.0

    return cost2

def one_hop_similarity_matrix(A, B, gtA, gtB):
    """
    Vectorized 1-hop similarity based on anchor matches.
    A, B: adjacency matrices (torch.Tensor)
    gtA, gtB: anchor index lists
    """
    nA, nB = A.shape[0], B.shape[0]
    device = A.device

    # Anchor matrix M
    M = torch.zeros((nA, nB), dtype=torch.float64, device=A.device)
    M[gtA, gtB] = 1.0
    
    # Compute number of matched anchored neighbors
    C = torch.mm(A, torch.mm(M, B.T))   # shape (nA, nB)
    maskA = torch.zeros(nA, dtype=torch.float64, device=device)
    maskA[gtA] = 1.0
    degA_gt = A @ maskA.unsqueeze(1)
    maskB = torch.zeros(nB, dtype=torch.float64, device=device)
    maskB[gtB] = 1.0
    degB_gt = B @ maskB.unsqueeze(1)
    # Degrees
    denom = degA_gt + degB_gt.T  # shape (nA, nB)

    
    # Similarity matrix
    sim = torch.zeros_like(C)
    sim[denom == 0] = 1.0
    sim[denom != 0] = 2 * C[denom != 0] / denom[denom != 0]
    #sim =-2*C
    
    return sim
def synchronize_graphs(G, G_Q, anchors_G, anchors_G_Q):
    """
    Create new graphs G1 and G1_Q where edges between aligned anchors
    are synchronized across G and G_Q.
    
    Parameters
    ----------
    G : nx.Graph
        Original graph G
    G_Q : nx.Graph
        Original graph G_Q
    anchors_G : list[int]
        List of nodes in G (anchors)
    anchors_G_Q : list[int]
        List of corresponding nodes in G_Q (same length as anchors_G)
    
    Returns
    -------
    G1, G1_Q : nx.Graph
        Synchronized graphs
    """

    # Build mapping dicts between anchors
    G_to_GQ = dict(zip(anchors_G, anchors_G_Q))
    GQ_to_G = dict(zip(anchors_G_Q, anchors_G))

    # Start with copies of the original graphs
    G1 = G.copy()
    G1_Q = G_Q.copy()
    counter=0
    counter1=0
    anchor_nodes = set(G_to_GQ.keys())
    anchor_edges = [(u, v) for u, v in G.edges() if u in anchor_nodes and v in anchor_nodes]
    print("Number of edges among anchors in G:", len(anchor_edges))

    # --- Step 1: Transfer edges from G to G_Q ---
    for u, v in G.edges():
        if u in G_to_GQ and v in G_to_GQ:
            u_q, v_q = G_to_GQ[u], G_to_GQ[v]
            if not G1_Q.has_edge(u_q, v_q):
                G1_Q.add_edge(u_q, v_q)
                counter=counter+1
            else:
                counter1=counter1+1

    # --- Step 2: Transfer edges from G_Q to G ---
    for u_q, v_q in G_Q.edges():
        if u_q in GQ_to_G and v_q in GQ_to_G:
            u, v = GQ_to_G[u_q], GQ_to_G[v_q]
            if not G1.has_edge(u, v):
                G1.add_edge(u, v)
                counter=counter+1
            else:
                counter1=counter1+1
    return G1, G1_Q
def one_hop_cost_neighbors(A, B, gtA, gtB):
    nA, nB = A.shape[0], B.shape[0]
    device = A.device

    # Anchor mapping matrix
    M = torch.zeros((nA, nB), dtype=torch.float64, device=device)
    M[gtA, gtB] = 1.0

    # Count matched anchored neighbors
    C = torch.mm(A, torch.mm(M, B.T))  # # of matched anchored neighbors
    # Total anchored neighbors for i and j
    maskA = torch.zeros(nA, dtype=torch.float64, device=device)
    maskA[gtA] = 1.0
    degA_gt = A @ maskA.unsqueeze(1)
    maskB = torch.zeros(nB, dtype=torch.float64, device=device)
    maskB[gtB] = 1.0
    degB_gt = B @ maskB.unsqueeze(1)
    # Cost = fraction of mismatched anchored neighbors
    denom = degA_gt + degB_gt.T
    cost = torch.zeros_like(C)
    nonzero = denom != 0
    w = torch.log1p(denom) / torch.log1p(denom.max())
    #cost[nonzero] = (degA_gt + degB_gt.T - 2*C)[nonzero] / denom[nonzero]
    cost[nonzero] = w[nonzero] * ((degA_gt + degB_gt.T - 2*C)[nonzero] / denom[nonzero])

    # If both nodes have no anchored neighbors, cost = 0
    cost[denom == 0] = 0.0
    return cost
def two_hop_similarity_matrix(A, B, gtA, gtB):

    """
    Vectorized 2-hop similarity based on ground-truth anchor matches.
    Only neighbors that have a ground-truth correspondence are counted.

    Args:
        A, B: adjacency matrices (torch.Tensor, shape nA x nA and nB x nB)
        gtA, gtB: lists of ground-truth anchor indices

    Returns:
        sim: similarity matrix (nA x nB)
    """
    nA, nB = A.shape[0], B.shape[0]
    device = A.device

    # Compute 2-hop adjacency matrices (binary)
    A2 = ((A + A @ A) > 0).double()
    B2 = ((B + B @ B) > 0).double()
    # Anchor mapping matrix
    M = torch.zeros((nA, nB), dtype=torch.float64, device=device)
    M[gtA, gtB] = 1.0

    # Count matched anchored 2-hop neighbors
    C = A2 @ M @ B2.T  # shape (nA, nB)

    # Precompute degrees of neighbors that can participate in ground-truth matching
    maskA = torch.zeros(nA, dtype=torch.float64, device=device)
    maskA[gtA] = 1.0
    degA_gt = A2 @ maskA.unsqueeze(1)  # shape (nA, 1)

    maskB = torch.zeros(nB, dtype=torch.float64, device=device)
    maskB[gtB] = 1.0
    degB_gt = B2 @ maskB.unsqueeze(1)  # shape (nB, 1)

    # Denominator: sum of mappable neighbors
    denom = degA_gt + degB_gt.T  # shape (nA, nB)

    # Similarity matrix
    sim = torch.zeros_like(C)
    sim[denom == 0] = 1.0
    sim[denom != 0] = 2 * C[denom != 0] / denom[denom != 0]

    return sim
iters=1
tun=[1,10,11,12]
tuns=["Alpine","Grad","JOENA","SlotAlign","NextAlign"]
tun=[13]
tuns=["NextAlign"]
nL=["testing"]
foldernames=['douban','allmv_tmdb','acm_dblp','fb_tw','ppi','cora','foursquare','phone']
n_G2 = [1118,5712,9872,1043,1767,2708,5120,1000] #s
n_G=[3906,6010,9916,1043,1767,2708,5313,1003] #t
gt_size=[1118,5174,6325,1043,1767,2708,1609,1000]
attrN=[True,True,True,False,True,True,False,False]
AlpiS=[0.1, 0.1, 0.1, 1.0,  0.1,  0.1,1.0,4]
ratio=0.20
#douban
foldernames=['phone']
n_G2 = [1000] #s
n_G=[1003] #t
gt_size=[1000]
attrN=[False]
for k in range(0,len(foldernames)):
        #G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        G = read_real_graph(n = n_G[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_edge.txt')
        print(G)
        DGS=G.number_of_nodes()
    # Get the number of edges
        DGES = G.number_of_edges()       
        for _ in nL: 
        #for noiseL in nL: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}/{ptun}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}/{ptun}'
                file_A_results = open(f'{folder1}/Thesis_results.txt', 'w')
                file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
                if (foldernames[k]=="foursquare" or foldernames[k]=="phone"):
                        F1=np.zeros((n_G2[k],1))
                        F2=np.zeros((n_G[k],1))
                else:
                    F2 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_feat.txt', dtype=float)  # shape: (n1, k)
                    F1 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_feat.txt', dtype=float)  # shape: (n2, k)
                #diffs = B_feat[:, None, :] - A_feat[None, :, :]   # shape: (n1, n2, k)
                #X = np.abs(diffs).sum(axis=2)  # shape: (n1, n2)
                #Feat = np.linalg.norm(diffs, axis=2)  # shape: (n1, n2)
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
                F2=F2
                F1=F1
                

# Split into two arrays
                if (foldernames[k]=="douban"):
                    csv2 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr1.csv", header=None).iloc[:, 1:].to_numpy()
                    csv1 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr2.csv", header=None).iloc[:, 1:].to_numpy()
                if (foldernames[k]=="acm_dblp"):
                    data = np.load(f'JOENA/datasets/ACM-DBLP_0.2.npz')
                    csv2=data['x2']
                    csv1=data['x1']
                for iter in range(iters):
                    print("iter",iter)
                    if (foldernames[k]=="douban" or foldernames[k]=="acm_dblp"):
                        F2=csv2
                        F1=csv1
                    #you have to do that because the features have ID making them 
                    #giving ground truth information
                    if (foldernames[k]=="fb_tw"):
                        F2=F2*0
                        F1=F1*0
                        
                    data_GT = np.loadtxt(f"./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True_{ratio}_{iter}.txt", dtype=int)
                    #F2=csv2
                    #F1=csv1
                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph.txt'
                    file_nodes = f'{folder_}/nodes.txt'
                    #Q_real = read_list(file_nodes)
                    #G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
                    G_Q = read_real_graph(n = n_G2[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_edge.txt')
                    pairs = []
                    with open(f'./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True.txt', "r") as f:
                        for line in f:
                            a, b = line.strip().split()
                            pairs.append((int(a), int(b)))

# ✅ Find the max node ID in each graph
                    max_A = n_G2[k]#max(a for a, b in pairs)
                    max_B = n_G2[k]#max(b for a, b in pairs)

# 2️⃣ Build A→B mapping with -1 for missing
                    true1=False
                    true2=False
                    #if douban/dblp/fb_tw no+1 -allmv_tmdbwith +1
                    if (foldernames[k]=="allmv_tmdb"):
                        max_A=max_A+1
                        max_B=max_B+1
                    A_to_B = [-1] * (max_A)
                    for a, b in pairs:
                        if (a>=max_A):
                            true1=True
                        else:
                         A_to_B[a] = b

# 3️⃣ Build B→A mapping with -1 for missing
                    B_to_A = [-1] * (max_B)
                    for a, b in pairs:
                        if (b>=max_B):
                            true2=True
                        else:
                            B_to_A[b] = a
                    print(true1,true2)
                    #A = nx.adjacency_matrix(G_Q).todense()
                    QGS=G_Q.number_of_nodes()
                    QGES = G_Q.number_of_edges()
                    anchors_G = data_GT[:, 1].tolist()
                    anchors_GQ = data_GT[:, 0].tolist()
                    #G1,G1_Q=synchronize_graphs(G,G_Q,anchors_G,anchors_GQ)
                    print(max(anchors_G)," value")
                    print(max(anchors_GQ)," value")
                    print(np.shape(F2))
                    print("G_Q",G_Q.number_of_edges())
                    print("G",G.number_of_edges())
                    if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb" or foldernames[k]=="foursquare" or foldernames[k]=="phone"):
                        anchors_G = data_GT[:, 0].tolist()
                        anchors_GQ = data_GT[:, 1].tolist()

                    G1,G1_Q=seed_link(anchors_G,anchors_GQ,G,G_Q)
                    if (foldernames[k]!="foursquare" and foldernames[k]!="phone"):
                        
                        F2_n,F1_n=synchronize_features(F2,F1,anchors_G,anchors_GQ)
                    else:
                        F2_n=F2
                        F1_n=F1
                        #F2_n*=0
                    #F1_n*=0
                    start = time.time()
                    #compare_features(F1,F2,A_to_B)
                    #compare_features(F1,F2,B_to_A)
                    if(tun[ptun]==1):
                        print("Alpine")
                        mun=0.1
                        #if(foldernames[k]=="fb_tw" or foldernames[k]=="foursquare" or foldernames[k]=="phone"):
                        if(attrN[k]==False):
                            mun=1
                        _, list_of_nodes, forb_norm = Alpine_supervised(G1_Q.copy(), G1.copy(),F1_n,F2_n,anchors_GQ,anchors_G,mun,weight=2)                    
                    elif(tun[ptun]==10):
                        print("GradAlignP")
                        list_of_nodes, forb_norm = gradPMain(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),anchors_GQ=anchors_GQ,anchors_G=anchors_G)
                    elif(tun[ptun]==11):
                        forb_norm=1
                        print("JOENA")
                        if foldernames[k] in ["acm_dblp","ppi","cora"]:
                            print("in")
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                            #data_GT1=data_GT
                            
                        else:
                            data_GT1=data_GT
                            
                        similarity = JOENA(foldernames[k],ratio,attrN[k],data_GT1)
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity.T
                        
                        #cora,douban,fb-tw,phone,allmv_tmdb,ppi  works
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    
                    elif(tun[ptun]==12):
                        forb_norm=1
                        print("SlotaAlign")
                        similarity = SlotaA(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),foldernames[k])
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    elif(tun[ptun]==13):
                        forb_norm=1
                        print("NextAlign")
                        if foldernames[k] not in ["acm_dblp","ppi","cora"]:
                            print("in")
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                            #data_GT1=data_GT
                            
                        else:
                            data_GT1=data_GT
                            
                        similarity = NextAlign(foldernames[k],ratio,attrN[k],data_GT1)
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity
                        
                        #cora,douban,fb-tw,phone,allmv_tmdb,ppi  works
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    
                    
                    end = time.time()
                    subgraph = G.subgraph(list_of_nodes)
                    PGS=subgraph.number_of_nodes()
                    PGES = subgraph.number_of_edges()
                    isomorphic=False
                    if(forb_norm==0):
                        isomorphic=True
                    time_diff = end - start
                    file_nodes_pred = open(f'{folder1_}/{tuns[ptun]}.txt','w')
                    for node in list_of_nodes: file_nodes_pred.write(f'{node}\n')
                    spec_norm=0
                    #accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
                    accuracy1 = np.sum(np.array(A_to_B)==np.array(list_of_nodes))/gt_size[0]
                    accuracy2 = np.sum(np.array(B_to_A)==np.array(list_of_nodes))/gt_size[0]
                    accuracy=0
                    if ({foldernames[k]}=="douban" or{foldernames[k]}=="allmv_tmdb" ):
                        accuracy=accuracy2
                    else:
                        accuracy=accuracy1
                    print("ACC 1 or 2?",accuracy1,accuracy2)
                    print(np.sum(np.array(A_to_B)==np.array(list_of_nodes)))
                    with open("differences.txt", "w") as f:
                        f.write("Differences A_to_B:\n")
                        f.write("\n\nDifferences B_to_A:\n")
                        f.write("\n\nAccuracy A_to_B: {:.4f}\n".format(accuracy1))
                        f.write("Accuracy B_to_A: {:.4f}\n".format(accuracy2))
                    file_A_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy1} {accuracy2} {time_diff} {isomorphic}\n')
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,isomorphic)          
            print('\n')
        print('\n\n')

